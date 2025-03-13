from pydantic import BaseModel

class ImageOCRRequest(BaseModel):
    image_url: str
    with_blocks: bool = False

class ImageOCRRequestFile(BaseModel):
    with_blocks: bool = False

class InferenceURLRequest(BaseModel):
    image_url: str
    text: str

class InferenceFileRequest(BaseModel):
    text: str
