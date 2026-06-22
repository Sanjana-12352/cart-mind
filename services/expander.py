import os
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List, Optional
class Category(BaseModel):
    label: str = Field(description="Specific product category name. Example: 'Balloons' not 'Decorations'")
    icon: str = Field(description="Single relevant emoji")

class ExpandedQuery(BaseModel):
    categories: List[Category] = Field(
        description="All categories needed. Minimum 4, maximum 10. Each must be a specific product type."
    )
DOMAIN_PROMPTS = {
    "fashion": "Expert fashion stylist. Think: specific clothing type, footwear type, bag type, jewellery type, beauty product.",
    "home_setup": "Expert interior designer. Think: specific furniture pieces, specific lighting, specific textiles, specific decor items for the exact space mentioned.",
    "electronics": "Expert tech consultant. Think: specific devices, specific peripherals, specific accessories, specific cables.",
    "beauty": "Expert beauty consultant. Think: specific skincare products, specific makeup items, specific tools.",
    "baby": "Expert baby specialist. Think: specific baby furniture, specific feeding items, specific clothing, specific safety items.",
    "fitness": "Expert fitness trainer. Think: specific equipment, specific clothing, specific supplements, specific accessories.",
    "office": "Expert workspace designer. Think: specific desk type, specific chair type, specific monitor, specific peripherals, specific storage.",
    "party": "Expert event planner. Think: specific decoration items, specific tableware, specific food items, specific games, specific gifts.",
    "stationery": "Expert stationery specialist. Think: specific pen types, specific paper types, specific notebook types, specific organizers.",
    "food": "Expert grocery specialist. Think: specific food categories, specific beverages, specific ingredients.",
    "general": "Expert personal shopper. Think of all specific product types needed for this goal."
}
def get_llm():
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=1000
    )
 
 
async def expand_query(
    query: str,
    domain: str,
    occasion: Optional[str] = None,
    persona: Optional[str] = None,
    quantity: Optional[int] = None
) -> dict:
 
    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=ExpandedQuery)
 
    domain_expertise = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["general"])
 
    system_content = (
        f"You are an {domain_expertise} "
        "Your job is to identify ALL specific product categories needed for a shopping goal. "
        "\n\n"
        "CRITICAL RULES: "
        "1. Return ONLY valid JSON with a 'categories' key "
        "2. Each category needs 'label' and 'icon' "
        "3. Labels must be SPECIFIC product names "
        "   'Balloons' not 'Decorations' "
        "   'Dining Table' not 'Furniture' "
        "   'Gel Pens' not 'Writing Instruments' "
        "4. Minimum 4 categories, maximum 10 "
        "5. Even simple queries get full expansion "
        "   'pen' → Ballpoint Pens, Gel Pens, Fountain Pens, Markers, Highlighters "
        "6. Consider quantity if mentioned "
        "   '30 people' → bulk/pack sizes in labels "
        "7. Return ONLY JSON, no text, no explanation "
    )
 
    human_content = (
        f"Shopping Goal: {query}\n"
        f"Domain: {domain}\n"
        f"Occasion: {occasion if occasion else 'not specified'}\n"
        f"Persona: {persona if persona else 'not specified'}\n"
        f"Quantity: {quantity if quantity else 'not specified'}\n\n"
        "Return JSON with all specific product categories needed."
    )
 
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content)
    ]
 
    response = await llm.ainvoke(messages)
    result = parser.parse(response.content)
    print(f"Expanded into {len(result.get('categories', []))} categories")
    return result

