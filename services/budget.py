import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class BudgetAllocation(BaseModel):
    allocations: Dict[str, float] = Field(
        description="A dictionary mapping each category label to its allocated budget amount. All values must add up to the total budget."
    )
    reasoning: str = Field(
        description="One line explanation of how budget was distributed."
    )


def get_llm():
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )


async def distribute_budget(
    budget: float,
    domain: str,
    categories: List[str]
) -> dict:

    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=BudgetAllocation)

    categories_str = ", ".join(categories)

    system_content = (
        "You are a budget planner for shopping. "
        "Distribute a total budget across shopping categories intelligently. "
        "Primary items get more budget, accessories get less. "
        "Return ONLY a valid JSON object. No text, no markdown, no explanation. "
        "The JSON must have 'allocations' (dict of category to amount) "
        "and 'reasoning' (one line string). "
        "All allocation values must add up exactly to the total budget."
    )

    human_content = (
        f"Total Budget: {budget} INR\n"
        f"Shopping Domain: {domain}\n"
        f"Categories: {categories_str}\n\n"
        "Return JSON with budget allocated per category. "
        "All allocations must sum to exactly the total budget."
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content)
    ]

    response = await llm.ainvoke(messages)
    result = parser.parse(response.content)

    return result


def get_category_budget(
    budget_result: dict,
    category_label: str,
    fallback_budget: Optional[float] = None
) -> Optional[float]:

    if not budget_result:
        return fallback_budget

    allocations = budget_result.get("allocations", {})

    if category_label in allocations:
        return allocations[category_label]

    for key, value in allocations.items():
        if key.lower() == category_label.lower():
            return value

    return fallback_budget