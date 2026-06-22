from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    query: str


class IntentOutput(BaseModel):
    domain: str
    occasion: Optional[str] = None
    budget: Optional[float] = None
    currency: Optional[str] = "INR"
    persona: Optional[str] = None
    quantity: Optional[int] = None


class Category(BaseModel):
    label: str
    icon: Optional[str] = "🛍️"
    products: List[str] = []


class QueryResponse(BaseModel):
    query: str
    intent: IntentOutput
    labels: List[str]
    categories: List[Category]