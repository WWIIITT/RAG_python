from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import MarkdownifyTransformer


async def load_markdown(url: str):
    loader = AsyncHtmlLoader(url)
    docs_html = await loader.aload()
    markdown_transformer = MarkdownifyTransformer()
    return markdown_transformer.transform_documents(docs_html)
