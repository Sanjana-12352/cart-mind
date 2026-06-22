from pydantic import BaseModel
from typing import List, Optional, Any


class QueryRequest(BaseModel):
    query: str


class ClassifyResponse(BaseModel):
    pipeline: Any
    routing: str
    zero_result_rescue_triggered: bool
    ui: Any