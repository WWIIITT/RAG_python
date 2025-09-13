from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase, Driver

from app.config import get_settings


_driver: Optional[Driver] = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        settings = get_settings()
        _driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def setup_vector_index() -> None:
    driver = get_driver()
    with driver.session(database=get_settings().neo4j_database) as session:
        session.run(
            """
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS { indexConfig: {
                `vector.dimensions`: 3072,
                `vector.similarity_function`: 'cosine'
            }}
            """
        )
        session.run(
            """
            CREATE FULLTEXT INDEX chunk_text_index IF NOT EXISTS
            FOR (c:Chunk) ON EACH [c.text]
            """
        )


# --- Query helpers ---

def find_document_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    driver = get_driver()
    with driver.session(database=get_settings().neo4j_database) as session:
        result = session.run(
            "MATCH (d:Document {hash: $hash}) RETURN d.id AS id",
            hash=file_hash,
        )
        record = result.single()
        return {"id": record["id"]} if record else None


def create_graph_from_document(document: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    driver = get_driver()
    with driver.session(database=get_settings().neo4j_database) as session:
        def tx_work(tx):
            create_nodes_query = """
                MERGE (d:Document {hash: $document.hash})
                ON CREATE SET d.name = $document.name, d.size = $document.size, d.mimetype = $document.mimetype, d.createdAt = timestamp(), d.id = randomUUID()
                WITH d
                UNWIND $chunks AS chunkData
                CREATE (c:Chunk {
                    text: chunkData.text,
                    pageNumber: chunkData.metadata.pageNumber,
                    embedding: chunkData.embedding,
                    id: randomUUID()
                })
                CREATE (d)-[:HAS_CHUNK]->(c)
                WITH d, c, chunkData.entities AS entities
                UNWIND entities AS entityData
                CALL apoc.merge.node([entityData.type], {name: entityData.name}) YIELD node AS e
                MERGE (c)-[:MENTIONS]->(e)
                WITH DISTINCT d
                RETURN d.id AS fileId
            """
            nodes_res = tx.run(create_nodes_query, document=document, chunks=chunks)
            record = nodes_res.single()
            if not record:
                fallback = tx.run("MATCH (d:Document {hash: $hash}) RETURN d.id AS fileId", hash=document["hash"]).single()
                if not fallback:
                    raise RuntimeError("無法在資料庫中建立或找到文件節點")
                file_id = fallback["fileId"]
            else:
                file_id = record["fileId"]

            create_rels_query = """
                UNWIND $chunks AS chunkData
                WITH chunkData WHERE size(chunkData.relationships) > 0
                UNWIND chunkData.relationships AS relData
                MATCH (sourceNode {name: relData.source})
                MATCH (targetNode {name: relData.target})
                CALL apoc.merge.relationship(sourceNode, relData.type, {}, {}, targetNode) YIELD rel
                RETURN count(rel) AS createdRels
            """
            tx.run(create_rels_query, chunks=chunks)
            return {"fileId": file_id}

        result = session.execute_write(tx_work)
        return result


def retrieve_graph_context(query_vector: List[float], k: int = 10, selected_file_ids: List[str] = []) -> List[Dict[str, Any]]:
    driver = get_driver()
    with driver.session(database=get_settings().neo4j_database) as session:
        result = session.run(
            """
            MATCH (doc:Document)-[:HAS_CHUNK]->(chunk)
            WHERE doc.id IN $selectedFileIds
            WITH doc, chunk, vector.similarity.cosine($queryVector, chunk.embedding) AS score
            ORDER BY score DESC
            LIMIT $k
            OPTIONAL MATCH (chunk)-[:MENTIONS]->(entity)
            RETURN chunk.text AS text , score, doc.name AS source, doc.id AS fileId, chunk.pageNumber AS page, chunk.id AS chunkId,
                   collect(CASE WHEN entity IS NOT NULL THEN {name: entity.name, type: labels(entity)[0]} ELSE null END) AS mentionedEntities
            ORDER BY score DESC
            """,
            queryVector=query_vector,
            k=k,
            selectedFileIds=selected_file_ids,
        )
        return [
            {
                "text": r["text"],
                "score": r["score"],
                "source": r["source"],
                "page": r["page"],
                "fileId": r["fileId"],
                "chunkId": r["chunkId"],
                "mentionedEntities": [e for e in r["mentionedEntities"] if e],
            }
            for r in result
        ]


def retrieve_context_by_entities(entity_names: List[str], selected_file_ids: List[str] = []) -> List[Dict[str, Any]]:
    if not entity_names:
        return []
    driver = get_driver()
    with driver.session(database=get_settings().neo4j_database) as session:
        result = session.run(
            """
            WITH [name IN $entityNames | toLower(name)] AS lowerCaseNames
            MATCH (entity) WHERE toLower(entity.name) IN lowerCaseNames
            MATCH (doc:Document)-[:HAS_CHUNK]->(chunk:Chunk)-[:MENTIONS]->(entity)
            WHERE size($selectedFileIds) = 0 OR doc.id IN $selectedFileIds
            WITH doc, chunk, collect(DISTINCT entity) AS mentionedQueryEntities
            RETURN
                chunk.text AS text,
                size(mentionedQueryEntities) AS score,
                doc.name AS source,
                doc.id AS fileId,
                chunk.pageNumber AS page,
                chunk.id AS chunkId,
                [e IN mentionedQueryEntities | {name: e.name, type: labels(e)[0]}] AS mentionedEntities
            ORDER BY page
            """,
            entityNames=entity_names,
            selectedFileIds=selected_file_ids,
        )
        return [
            {
                "text": r["text"],
                "score": r["score"],
                "source": r["source"],
                "page": r["page"],
                "fileId": r["fileId"],
                "chunkId": r["chunkId"],
                "mentionedEntities": r["mentionedEntities"],
            }
            for r in result
        ]


def retrieve_context_by_keywords(keywords: str, selected_file_ids: List[str] = [], k: int = 10) -> List[Dict[str, Any]]:
    driver = get_driver()
    with driver.session(database=get_settings().neo4j_database) as session:
        result = session.run(
            """
            CALL db.index.fulltext.queryNodes("chunk_text_index", $keywords) YIELD node AS chunk, score
            MATCH (doc:Document)-[:HAS_CHUNK]->(chunk)
            WHERE size($selectedFileIds) = 0 OR doc.id IN $selectedFileIds
            OPTIONAL MATCH (chunk)-[:MENTIONS]->(entity)
            RETURN
                chunk.text AS text,
                score,
                doc.name AS source,
                doc.id AS fileId,
                chunk.pageNumber AS page,
                chunk.id AS chunkId,
                collect(CASE WHEN entity IS NOT NULL THEN {name: entity.name, type: labels(entity)[0]} ELSE null END) AS mentionedEntities
            ORDER BY score DESC
            LIMIT $k
            """,
            keywords=keywords,
            selectedFileIds=selected_file_ids,
            k=k,
        )
        return [
            {
                "text": r["text"],
                "score": r["score"],
                "source": r["source"],
                "page": r["page"],
                "fileId": r["fileId"],
                "chunkId": r["chunkId"],
                "mentionedEntities": [e for e in r["mentionedEntities"] if e],
            }
            for r in result
        ]


def get_files_list() -> List[Dict[str, Any]]:
    driver = get_driver()
    with driver.session(database=get_settings().neo4j_database) as session:
        result = session.run(
            """
            MATCH (d:Document)
            WHERE d.id IS NOT NULL AND d.hash IS NOT NULL
            WITH d
            ORDER BY d.createdAt DESC
            RETURN 
                d.id AS id,
                d.name AS name,
                d.size AS size,
                d.mimetype AS mime_type,
                d.createdAt AS upload_date,
                COUNT {(d)-[:HAS_CHUNK]->()} AS total_chunks
            """
        )
        files = []
        for record in result:
            created_at = record["upload_date"]
            files.append(
                {
                    "id": record["id"],
                    "filename": record["name"],
                    "original_name": record["name"],
                    "file_size": record["size"],
                    "mime_type": record["mime_type"],
                    "upload_date": str(created_at) if created_at else None,
                    "status": "completed",
                    "total_chunks": int(record["total_chunks"]),
                }
            )
        return files


def delete_file(file_id: str) -> Dict[str, Any]:
    driver = get_driver()
    with driver.session(database=get_settings().neo4j_database) as session:
        tx = session.begin_transaction()
        try:
            file_info = tx.run("MATCH (d:Document {id: $fileId}) RETURN d", fileId=file_id)
            record = file_info.single()
            if not record:
                raise RuntimeError("檔案不存在")
            deleted_props = record["d"]._properties  # type: ignore
            # 1) 刪除該文件的所有 Chunk（及其 MENTIONS 關係）
            tx.run(
                """
                MATCH (d:Document {id: $fileId})-[:HAS_CHUNK]->(c:Chunk)
                DETACH DELETE c
                """,
                fileId=file_id,
            )
            # 2) 刪除文件節點本身
            tx.run(
                """
                MATCH (d:Document {id: $fileId})
                DETACH DELETE d
                """,
                fileId=file_id,
            )
            # 3) 清理已失去支撐的實體-實體關係（沒有任何同一 chunk 同時提及兩端點的關係）
            tx.run(
                """
                MATCH (a)-[r]->(b)
                WHERE type(r) <> 'HAS_CHUNK' AND type(r) <> 'MENTIONS'
                AND NOT EXISTS {
                    MATCH (c:Chunk)-[:MENTIONS]->(a)
                    WITH c
                    MATCH (c)-[:MENTIONS]->(b)
                }
                DELETE r
                """
            )
            # 4) 清理不再被任何 Chunk 提及的孤立實體
            tx.run(
                """
                MATCH (e)
                WHERE NOT e:Document AND NOT e:Chunk
                AND NOT EXISTS { MATCH (:Chunk)-[:MENTIONS]->(e) }
                DETACH DELETE e
                """
            )
            tx.commit()
            return {
                "message": f"檔案 '{deleted_props.get('name')}' 已成功從資料庫刪除。",
                "deletedFile": {"id": deleted_props.get("id"), "name": deleted_props.get("name")},
            }
        except Exception:
            tx.rollback()
            raise


def get_specific_file(file_id: str) -> Dict[str, Any]:
    driver = get_driver()
    with driver.session(database=get_settings().neo4j_database) as session:
        result = session.run(
            """
            MATCH (d:Document {id: $fileId})
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            WHERE d.id IS NOT NULL AND d.hash IS NOT NULL
            WITH d, c 
            ORDER BY c.pageNumber ASC
            RETURN d, collect({
                id: c.id,
                content: c.text,
                pageNumber: c.pageNumber
            }) AS chunks
            """,
            fileId=file_id,
        )
        record = result.single()
        if not record or not record["d"]:
            raise RuntimeError("檔案不存在")
        file_node = record["d"]._properties  # type: ignore
        raw_chunks = [c for c in record["chunks"] if c.get("id")]
        formatted_file = {
            "id": file_node.get("id"),
            "filename": file_node.get("name"),
            "original_name": file_node.get("name"),
            "file_size": str(file_node.get("size")),
            "mime_type": file_node.get("mimetype"),
            "upload_date": str(file_node.get("createdAt")) if file_node.get("createdAt") else None,
            "status": "completed",
            "total_chunks": len(raw_chunks),
        }
        formatted_chunks = [
            {"id": c.get("id"), "content": c.get("content"), "chunk_index": c.get("pageNumber")}
            for c in raw_chunks
        ]
        return {"file": formatted_file, "chunks": formatted_chunks}
