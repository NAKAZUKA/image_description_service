from pydantic import BaseModel
from typing import List, Optional

class Block(BaseModel):
    block: str
    text: str

class OCRResponse(BaseModel):
    text: Optional[str] = None
    blocks: Optional[List[Block]] = None

class InferenceResponse(BaseModel):
    content: str

class StreamingResponse(BaseModel):
    chunk: str