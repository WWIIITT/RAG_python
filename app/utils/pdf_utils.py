from typing import List
from pypdf import PdfReader
from io import BytesIO


async def extract_text_by_page(buffer: bytes) -> List[str]:
    reader = PdfReader(BytesIO(buffer))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return pages
