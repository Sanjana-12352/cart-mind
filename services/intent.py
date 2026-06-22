import os
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Optional


class IntentOutput(BaseModel):
    domain: str = Field(
        description="EXACTLY one of: fashion, home_setup, electronics, beauty, baby, fitness, office, party, stationery, food, general"
    )
    occasion: Optional[str] = Field(
        default=None,
        description="Specific occasion if mentioned. Null if not mentioned."
    )
    budget: Optional[float] = Field(
        default=None,
        description="Budget as number only. Null if not mentioned."
    )
    currency: Optional[str] = Field(
        default="INR",
        description="INR by default. USD only if dollar or $ mentioned."
    )
    persona: Optional[str] = Field(
        default=None,
        description="Type of person if mentioned. Null if not mentioned."
    )
    quantity: Optional[int] = Field(
        default=None,
        description="Number of people or units as INTEGER only. 30 new hires = 30. party for 10 = 10. NEVER return words like 'multiple' or 'several'. Return null if exact number not mentioned."
    )


async def classify_intent(query: str) -> dict:

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=500
    )

    parser = JsonOutputParser(pydantic_object=IntentOutput)

    system_content = (
        "You are an expert e-commerce intent classifier. "
        "Analyze ANY shopping query no matter how simple or complex. "
        "Even single word queries like 'pen' or 'chair' must be classified. "
        "CRITICAL RULES: "
        "1. domain must be EXACTLY one of: "
        "fashion, home_setup, electronics, beauty, baby, "
        "fitness, office, party, stationery, food, general. "
        "2. Never return null for domain. Always classify. "
        "3. pen or pencil or notebook = stationery. "
        "4. chair or dining table or sofa = home_setup. "
        "5. office party or office event or holiday party = party NOT office. "
        "6. office desk setup or study desk = office. "
        "7. quantity MUST be an integer number or null. "
        "   NEVER return words like multiple, several, many, various. "
        "   If no specific number mentioned return null. "
        "8. Return ONLY valid JSON, nothing else."
    )

    human_content = (
        f"Classify this shopping query: '{query}'\n\n"
        "Return JSON with domain, occasion, budget, currency, persona, quantity."
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content)
    ]

    response = await llm.ainvoke(messages)
    result = parser.parse(response.content)

    print(f"Intent classified: {result}")
    return result