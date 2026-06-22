import os
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Optional
class IntentOutput(BaseModel):
    domain: str = Field(
    description="""
    EXACTLY one of these values:
        'fashion'     - clothing, outfit, accessories, jewellery, shoes
        'home_setup'  - furniture, dining, bedroom, living room, kitchen
        'electronics' - phones, laptops, gadgets, tech devices
        'beauty'      - skincare, makeup, haircare, wellness
        'baby'        - baby products, nursery, toys, baby clothing
        'fitness'     - gym, workout, sports, exercise
        'office'      - office desk setup, workspace, study desk ONLY
                        NOT office events or office parties
        'party'       - party planning, events, celebrations,
                        holiday parties, office parties, birthdays as events
        'stationery'  - pens, paper, notebooks, office supplies
        'food'        - food items, groceries, beverages, snacks
        'general'     - anything else
 
        EXAMPLES:
        'pen'                        → stationery
        'chair'                      → home_setup OR office (context based)
        'birthday party'             → party
        'office holiday party'       → party (NOT office)
        'study desk for student'     → office
        'dining table'               → home_setup
        'gaming laptop'              → electronics
        'skincare routine'           → beauty
        'home gym'                   → fitness
        'baby nursery'               → baby
        """
    )
    
 
    occasion: Optional[str] = Field(
        default=None,
        description="Specific occasion if mentioned. Examples: birthday, wedding, christmas, new hire, graduation. Null if not mentioned."
    )
 
    budget: Optional[float] = Field(
        default=None,
        description="Budget as number only. 'under 5k' → 5000. '1 lakh' → 100000. Null if not mentioned."
    )
 
    currency: Optional[str] = Field(
        default="INR",
        description="INR by default. USD only if $ or dollar mentioned."
    )
 
    persona: Optional[str] = Field(
        default=None,
        description="Type of person if mentioned. Examples: student, developer, new hire, child, athlete. Null if not mentioned."
    )
    quantity: Optional[int] = Field(
        default=None,
        description="Number of people or units if mentioned. '30 new hires' → 30. 'party for 10' → 10. Null if not mentioned."
    )
def get_llm():
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=500
    )
async def classify_intent(query: str) ->dict:
    llm=get_llm()
    parser=JsonOutputParser(pydantic_object=IntentOutput)
    system_content=(
        "You are an expert e-commerce intent classifier"
        "Analyze ANY shopping query no matter how simple or complex"
        "Even single word queries like'pen' or 'chair' must be classified. "
        "\n\n"
        "CRITICAL RULES: "
        "1. domain must be EXACTLY one of the allowed values "
        "2. Never return null for domain - always classify "
        "3. For simple queries like 'pen' → domain: stationery "
        "4. For simple queries like 'chair' → domain: home_setup "
        "5. office domain = ONLY desk/workspace setup "
        "   office PARTY or EVENT = party domain "
        "6. Extract ALL available context from the query "
        "7. Return ONLY valid JSON, nothing else "
    )

    human_content = (
        f"Classify this shopping query:\n\n'{query}'\n\n"
        "Return JSON with domain, occasion, budget, currency, persona, quantity."
    )
 
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content)
    ]
 
    response = await llm.ainvoke(messages)
    result = parser.parse(response.content)
 
    print(f" Intent classified: {result}")
    return result