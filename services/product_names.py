import os
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List
import asyncio
class CategoryProducts(BaseModel):
    label: str = Field(description="Category label exactly as provided")
    products: List[str] = Field(description="List of 5 to 10 specific product names for this category")
class AllProducts(BaseModel):
    categories: List[CategoryProducts] = Field(
        description="All categories with their product names"
    )
 
 
def get_llm():
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=2000
    )
 
 
async def generate_product_names(
    query: str,
    categories: List[dict],
    domain: str,
    occasion: str = None,
    quantity: int = None
) -> List[dict]:
 
    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=AllProducts)
 
    category_labels = [cat.get("label", "") for cat in categories]
    categories_str = "\n".join([f"- {label}" for label in category_labels])
 
    quantity_context = ""
    if quantity:
        quantity_context = f"Note: This is for {quantity} people. Include bulk/pack sizes where relevant."
 
    system_content = (
        "You are an expert product catalog specialist for an e-commerce platform. "
        "Generate realistic and specific product names for shopping categories. "
        "\n\n"
        "CRITICAL RULES: "
        "1. Return ONLY valid JSON with a 'categories' key "
        "2. Each category must have 'label' (exact match) and 'products' (list of strings) "
        "3. Generate 5 to 10 specific product names per category "
        "4. Product names must be realistic and specific "
        "   BAD:  'Balloon' "
        "   GOOD: 'Latex Balloon Pack of 50 Assorted Colors' "
        "5. Include size, quantity, material where relevant "
        "6. Product names should match what you find in real stores "
        "7. Consider the context and occasion "
        "8. Return ONLY JSON, no text, no explanation "
        f"{quantity_context}"
    )
 
    human_content = (
        f"Shopping Goal: {query}\n"
        f"Domain: {domain}\n"
        f"Occasion: {occasion if occasion else 'not specified'}\n\n"
        f"Generate product names for these categories:\n{categories_str}\n\n"
        "Return JSON with product names for each category."
    )
 
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content)
    ]
 
    response = await llm.ainvoke(messages)
    result = parser.parse(response.content)
 
    generated_categories = result.get("categories", [])
 
    final_categories = []
    for original_cat in categories:
        label = original_cat.get("label", "")
        icon = original_cat.get("icon", "🛍️")
 
        products = []
        for gen_cat in generated_categories:
            if gen_cat.get("label", "").lower() == label.lower():
                products = gen_cat.get("products", [])
                break
 
        final_categories.append({
            "label": label,
            "icon": icon,
            "products": products
        })
 
    print(f"✅Product names generated for {len(final_categories)} categories")
    return final_categories
 