from pydantic import BaseModel
from typing import List, Optional, Dict


class QueryRequest(BaseModel):
    query: str


class FilterRequest(BaseModel):
    min_price: Optional[float] = 0
    max_price: Optional[float] = 100000
    min_rating: Optional[float] = 0
    prime_only: Optional[bool] = False
    brand: Optional[str] = None


class Intent(BaseModel):
    domain: str
    occasion: Optional[str] = None
    budget: Optional[float] = None
    currency: Optional[str] = "INR"
    persona: Optional[str] = None


class Product(BaseModel):
    asin: str
    title: str
    price: float
    rating: Optional[float] = None
    total_reviews: Optional[int] = None
    image: str
    affiliate_url: str
    prime: Optional[bool] = False
    in_stock: Optional[bool] = True


class Category(BaseModel):
    label: str
    icon: Optional[str] = "🛍️"
    products: List[Product]


class QueryResponse(BaseModel):
    intent: Intent
    labels: List[str]
    budget_split: Optional[Dict[str, float]] = None
    categories: List[Category]


class ProductDetailResponse(BaseModel):
    asin: str
    title: str
    description: Optional[str] = None
    price: float
    rating: Optional[float] = None
    total_reviews: Optional[int] = None
    images: List[str]
    availability: Optional[str] = "In Stock"
    prime: Optional[bool] = False
    brand: Optional[str] = None
    category: Optional[str] = None
    affiliate_url: str
    similar_products: Optional[List[Product]] = []