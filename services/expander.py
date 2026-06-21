import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List, Optional


class CategoryItems(BaseModel):
    label: str = Field(description="The category name. Short and clear.")
    icon: str = Field(description="A single relevant emoji for this category.")
    search_terms: List[str] = Field(description="List of 3 to 5 optimized Amazon search terms.")


class ExpandedQuery(BaseModel):
    categories: List[CategoryItems] = Field(
        description="Complete list of ALL categories needed. Minimum 4, maximum 10."
    )


DOMAIN_PROMPTS = {
    "fashion": "You are an expert fashion stylist. List every item needed for a complete look including clothing, footwear, bags, jewellery, and beauty essentials.",
    "home_setup": "You are an expert interior designer. List every item needed to completely set up the space including furniture, decor, lighting, storage, linen, and kitchenware.",
    "electronics": "You are an expert tech consultant. List every item needed for a complete tech setup including primary device, peripherals, cables, storage, and accessories.",
    "office": "You are an expert workspace designer. List every item needed for a productive workspace including desk, chair, monitor, keyboard, mouse, lighting, and storage.",
    "beauty": "You are an expert beauty consultant. List every product needed for a complete routine including cleanser, toner, serum, moisturiser, SPF, makeup, and tools.",
    "baby": "You are an expert baby product specialist. List every item needed for a safe setup including furniture, bedding, feeding, bathing, clothing, toys, and safety items.",
    "fitness": "You are an expert fitness trainer. List every item needed for a complete workout setup including equipment, clothing, footwear, supplements, and accessories.",
    "general": "You are an expert personal shopper. List every item needed to completely fulfill the shopping goal grouped into logical categories."
}


def get_llm():
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY")
    )


async def expand_query(
    query: str,
    domain: str,
    occasion: Optional[str] = None,
    budget: Optional[float] = None,
    persona: Optional[str] = None
) -> dict:

    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=ExpandedQuery)

    domain_expertise = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["general"])

    system_content = (
        domain_expertise
        + "\n\nReturn a JSON object with a 'categories' key. "
        + "Each category must have 'label', 'icon', and 'search_terms'. "
        + "Minimum 4 categories, maximum 10. "
        + "Search terms must be specific Amazon ready queries for the Indian market. "
        + "Return ONLY valid JSON, nothing else."
    )

    human_content = (
        f"Shopping Goal: {query}\n"
        f"Domain: {domain}\n"
        f"Occasion: {occasion if occasion else 'not specified'}\n"
        f"Budget: {budget if budget else 'not specified'}\n"
        f"Persona: {persona if persona else 'not specified'}\n\n"
        "Return the JSON object with all categories needed."
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content)
    ]

    response = await llm.ainvoke(messages)
    result = parser.parse(response.content)

    print(f"RAW EXPANDER RESULT: {result}")
    return result