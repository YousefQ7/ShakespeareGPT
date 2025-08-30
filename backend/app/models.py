from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class GenerationRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 1.0
    top_k: Optional[int] = None
    max_new_tokens: Optional[int] = 100

class GenerationResponse(BaseModel):
    id: int
    prompt: str
    response: str
    temperature: Optional[float]
    top_k: Optional[int]
    max_new_tokens: Optional[int]
    created_at: datetime

class GenerationHistory(BaseModel):
    id: int
    prompt: str
    response: str
    temperature: Optional[float]
    top_k: Optional[int]
    max_new_tokens: Optional[int]
    created_at: datetime
