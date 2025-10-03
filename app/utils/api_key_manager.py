from typing import Optional
from app.config import get_settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


_keys = []
_index = 0


def _ensure_init():
    global _keys, _index
    if not _keys:
        settings = get_settings()
        api_list = (settings.google_api_key_list or "").split(",")
        _keys = [k.strip() for k in api_list if k.strip()]
        _index = 0


def reset_key_index() -> None:
    global _index
    _ensure_init()
    _index = 0


def get_keys_count() -> int:
    _ensure_init()
    return len(_keys)


def switch_to_next_key() -> bool:
    global _index
    _ensure_init()
    _index += 1
    return _index < len(_keys)


def get_embedding_model() -> Optional[GoogleGenerativeAIEmbeddings]:
    _ensure_init()
    # Avoid creating a client without an API key; return None to indicate embeddings are unavailable
    if not _keys or _index >= len(_keys):
        return None
    return GoogleGenerativeAIEmbeddings(
        model=get_settings().google_ai_embeddings or "gemini-embedding-001",
        google_api_key=_keys[_index],
    )


def get_query_entity_extraction_model() -> Optional[ChatGoogleGenerativeAI]:
    _ensure_init()
    # If no API key is configured, return None so callers can skip this step
    if not _keys:
        return None
    api_key = _keys[0]
    schema = {
        "title": "extract_entities_schema",
        "type": "object",
        "properties": {
            "entities": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["entities"],
    }
    return ChatGoogleGenerativeAI(model=get_settings().google_ai_model or "gemini-2.5-flash", api_key=api_key).with_structured_output(schema)


def get_graph_extraction_model() -> Optional[ChatGoogleGenerativeAI]:
    _ensure_init()
    if not _keys or _index >= len(_keys):
        return None
    api_key = _keys[_index]
    schema = {
        "title": "graph_extraction_schema",
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "type": {"type": "string"}},
                    "required": ["name", "type"],
                },
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"source": {"type": "string"}, "target": {"type": "string"}, "type": {"type": "string"}},
                    "required": ["source", "target", "type"],
                },
            },
        },
        "required": ["entities", "relationships"],
    }
    return ChatGoogleGenerativeAI(model=get_settings().google_ai_model or "gemini-2.5-flash", api_key=api_key).with_structured_output(schema)


def get_answer_generation_model(schema, options=None) -> Optional[ChatGoogleGenerativeAI]:
    _ensure_init()
    if not _keys or _index >= len(_keys):
        return None
    api_key = _keys[_index]
    temperature = (options or {}).get("temperature", 0.7)
    thinking_budget = (options or {}).get("thinking_budget", 0)  # Default: 0 for faster responses
    max_output_tokens = (options or {}).get("max_output_tokens", 2000)  # Default: 8192 tokens
    
    # Control thinking budget - lower value = faster responses (range: 1024-8192)
    # Set to 1024 for fastest, 8192 for most thoughtful responses
    model_kwargs = {
        "thinking_config": {
            "thinking_budget": thinking_budget
        }
    }
    
    base = ChatGoogleGenerativeAI(
        model=get_settings().google_ai_model or "gemini-2.5-flash", 
        api_key=api_key, 
        temperature=temperature,
        max_output_tokens=max_output_tokens,  # Control output length
        model_kwargs=model_kwargs
    )
    print(base)
    return base.with_structured_output(schema)
