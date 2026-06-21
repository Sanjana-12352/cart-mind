import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List, Optional


class OptimizedCategory(BaseModel):
    label: str = Field(description="The category label exactly as provided")
    optimized_terms: List[str] = Field(description="List of 3 highly optimized Amazon search terms for this category.")


class OptimizedTerms(BaseModel):
    categories: List[OptimizedCategory] = Field(
        description="All categories with their optimized search terms."
    )


def get_llm():
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )


async def optimize_search_terms(
    query: str,
    domain: str,
    categories: List[str],
    occasion: str = None,
    persona: str = None
) -> dict:

    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=OptimizedTerms)

    categories_formatted = "\n".join([
        f"{i+1}. {cat}" for i, cat in enumerate(categories)
    ])

    system_content = (
        "You are an Amazon search specialist. "
        "Convert category names into optimized Amazon search queries. "
        "Return ONLY a valid JSON object with a 'categories' key. "
        "Each category must have 'label' and 'optimized_terms'. "
        "Do NOT return any text, markdown, or explanation. ONLY JSON."
    )

    human_content = (
        f"Original Query: {query}\n"
        f"Domain: {domain}\n"
        f"Occasion: {occasion if occasion else 'not specified'}\n"
        f"Persona: {persona if persona else 'not specified'}\n\n"
        f"Categories to optimize:\n{categories_formatted}\n\n"
        "Return JSON with optimized Amazon search terms for each category. "
        "Each category needs exactly 3 specific search terms. "
        "Format: category name + material/type + use case + size/quantity where relevant."
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content)
    ]

    response = await llm.ainvoke(messages)
    result = parser.parse(response.content)

    return result


def get_terms_for_category(
    optimized_result: dict,
    category_label: str
) -> List[str]:

    categories = optimized_result.get("categories", [])

    for category in categories:
        if category.get("label") == category_label:
            return category.get("optimized_terms", [])
        if category.get("label", "").lower() == category_label.lower():
            return category.get("optimized_terms", [])

    return [category_label]