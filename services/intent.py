import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Optional
class IntentOutput(BaseModel):
    domain: str = Field(
    description="""
    The shopping domain this query belongs to.
    Must be EXACTLY one of these values:
    'fashion' - clothing, accessories, jewellery, shoes
    'home_setup' - furniture, kitchen, bedroom, dining, living room
    'electronics' - phones, laptops, gadgets, accessories
    'beauty' - skincare, makeup, haircare, wellness
    'baby' - baby products, nursery, toys, clothing
    'fitness' - gym equipment, sportswear, supplements
    'office' - office furniture, stationery, tech setup
    'general' - anything that doesn't fit above

    Return ONLY the exact string value from the list above.
    Do NOT return anything else like 'Home & Furniture'.
    """
)
 
    occasion: Optional[str] = Field(
        default=None,
        description="""
        The specific occasion or use case if mentioned.
        Examples: 'birthday', 'wedding', 'college', 'office',
        'vacation', 'gym', 'dinner party', 'baby shower'
        Return null if no specific occasion is mentioned.
        """
    )
 
    budget: Optional[float] = Field(
        default=None,
        description="""
        The maximum budget in numbers only if mentioned.
        Examples:
        'under 5k' → 5000
        'below 10000' → 10000
        'around 2000' → 2000
        '500 rupees' → 500
        Return null if no budget is mentioned.
        """
    )
 
    currency: Optional[str] = Field(
        default="INR",
        description="""
        The currency of the budget.
        Default to 'INR' for Indian Rupees.
        Change to 'USD' only if $ or dollar is mentioned.
        """
    )
 
    persona: Optional[str] = Field(
        default=None,
        description="""
        The type of person or their role if mentioned.
        Examples: 'developer', 'designer', 'student',
        'new mother', 'college student', 'gamer', 'athlete'
        Return null if no persona is mentioned.
        """
    )
def get_llm():
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
def get_intent_prompt():
    return ChatPromptTemplate.from_messages([
        (
    "system",
    """
    You are an expert e-commerce intent classifier
    for an AI shopping assistant called CartMind.

    Your job is to analyze a user's shopping query
    and extract structured information from it.

    CRITICAL RULES:
    1. domain must be EXACTLY one of:
       fashion, home_setup, electronics, beauty,
       baby, fitness, office, general
    2. Never invent new domain values
    3. Always return valid JSON
    4. Extract budget as a number only
    5. Return null for fields not mentioned
    """
),
        (
            "human",
            
            "Analyze this shopping query and extract intent:\n\n{query}"
        )
    ])
async def classify_intent(query: str) -> dict:
    """
    Takes a raw user query string
    Returns a dictionary with classified intent
 
    Example:
    Input:  "birthday outfit under 5k"
    Output: {
        "domain": "fashion",
        "occasion": "birthday",
        "budget": 5000.0,
        "currency": "INR",
        "persona": null
    }
    """
    llm = get_llm()
    prompt = get_intent_prompt()
    parser = JsonOutputParser(pydantic_object=IntentOutput)
    chain=prompt|llm|parser
    result=await chain.ainvoke({"query":query})
    return result

 