import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List, Optional


class CategoryItems(BaseModel):
    label: str = Field(description="SPECIFIC product category name. Never generic. Example: 'Dining Table' not 'Furniture'. 'Desk Lamp' not 'Lighting'. 'Stationery Organizer' not 'Storage'.")
    icon: str = Field(description="A single relevant emoji for this category.")
    search_terms: List[str] = Field(description="List of 3 to 5 optimized Amazon search terms specific to this category.")


class ExpandedQuery(BaseModel):
    categories: List[CategoryItems] = Field(
        description="Complete list of ALL specific categories needed. Minimum 4, maximum 10. Every label must be a specific product name, never a generic word."
    )


DOMAIN_PROMPTS = {
    "fashion": """
        You are an expert fashion stylist and personal shopper.
        List every item needed for a complete look.
        Use SPECIFIC labels:
        'Dress' not 'Clothing'
        'Block Heels' not 'Footwear'
        'Clutch Bag' not 'Bags'
        'Statement Earrings' not 'Jewellery'
        'Lip Gloss' not 'Beauty'
        Think: outfit, footwear, bags, jewellery, beauty essentials.
        Consider the occasion and style mentioned.
    """,

    "home_setup": """
        You are an expert interior designer and home stylist.
        List every item needed for the SPECIFIC space mentioned in the query.
        Use SPECIFIC labels based on the space:

        For DINING AREA:
        'Dining Table', 'Dining Chairs', 'Dinner Plates',
        'Glassware', 'Cutlery Set', 'Tablecloth',
        'Placemats', 'Pendant Light', 'Centerpiece'

        For BEDROOM:
        'Bed Frame', 'Mattress', 'Pillow Set',
        'Bedsheet Set', 'Wardrobe', 'Bedside Table',
        'Bedroom Lamp', 'Curtains', 'Mirror'

        For LIVING ROOM:
        'Sofa Set', 'Coffee Table', 'TV Unit',
        'Curtains', 'Area Rug', 'Cushion Set',
        'Wall Art', 'Floor Lamp', 'Bookshelf'

        For KITCHEN:
        'Cookware Set', 'Dinner Set', 'Storage Containers',
        'Kitchen Appliances', 'Knife Set', 'Cutting Board'

        NEVER use generic labels like 'Furniture', 'Decor',
        'Lighting', 'Storage', 'Linen', 'Kitchenware'.
        ALWAYS use specific product names as labels.
    """,

    "electronics": """
        You are an expert tech consultant and setup specialist.
        List every item needed for a complete tech setup.
        Use SPECIFIC labels:
        'Gaming Monitor' not 'Monitor'
        'Mechanical Keyboard' not 'Keyboard'
        'Wireless Mouse' not 'Mouse'
        'Noise Cancelling Headphones' not 'Headphones'
        Think: primary device, peripherals, cables,
        storage, stands, cooling, and accessories.
    """,

    "office": """
        You are an expert workspace and study setup specialist.
        List every item needed for the specific setup mentioned.
        Use SPECIFIC labels based on persona:

        For STUDENT STUDY DESK:
        'Study Table', 'Study Chair', 'Desk Lamp',
        'Stationery Organizer', 'Laptop Stand',
        'Headphones', 'Whiteboard', 'Notebook Set',
        'Pen Holder', 'Book Stand'

        For PROFESSIONAL HOME OFFICE:
        'Office Desk', 'Ergonomic Chair', 'Monitor',
        'Mechanical Keyboard', 'Wireless Mouse',
        'Webcam', 'Cable Management', 'Desk Organizer',
        'Monitor Stand', 'Ring Light'

        For DEVELOPER SETUP:
        'Dual Monitor Setup', 'Mechanical Keyboard',
        'Ergonomic Chair', 'Monitor Stand',
        'Webcam', 'USB Hub', 'Cable Management',
        'Desk Mat', 'External Hard Drive'

        NEVER use generic labels like 'Furniture',
        'Tech', 'Accessories', 'Storage'.
        ALWAYS use specific product names.
    """,

    "beauty": """
        You are an expert beauty consultant and skincare specialist.
        List every product needed for a complete routine.
        Use SPECIFIC labels:
        'Face Wash' not 'Cleanser'
        'Vitamin C Serum' not 'Serum'
        'Moisturiser SPF' not 'Moisturiser'
        'Kajal' not 'Eye Makeup'
        'Lipstick' not 'Lip Products'
        Think: cleanser, toner, serum, moisturiser,
        SPF, makeup, tools, and treatments.
        Consider skin type and concerns if mentioned.
    """,

    "baby": """
        You are an expert baby product specialist and nursery designer.
        List every item needed for a safe and complete baby setup.
        Use SPECIFIC labels:
        'Baby Crib' not 'Furniture'
        'Baby Monitor' not 'Electronics'
        'Feeding Bottle Set' not 'Feeding'
        'Baby Bath Tub' not 'Bathing'
        'Swaddle Blanket' not 'Linen'
        Think: furniture, bedding, feeding, bathing,
        clothing, toys, safety items, and care products.
        Safety is the top priority.
    """,

    "fitness": """
        You are an expert fitness trainer and gym setup specialist.
        List every item needed for a complete workout setup.
        Use SPECIFIC labels:
        'Yoga Mat' not 'Equipment'
        'Resistance Bands' not 'Accessories'
        'Dumbbell Set' not 'Weights'
        'Gym Shoes' not 'Footwear'
        'Protein Supplement' not 'Nutrition'
        Think: equipment, clothing, footwear,
        supplements, accessories, and recovery items.
    """,

    "party": """
        You are an expert event planner and party organizer.
        List every item needed for a successful party or event.
        Use SPECIFIC labels:
        'Party Decorations' not 'Decor'
        'Balloon Set' not 'Balloons'
        'Party Tableware' not 'Tableware'
        'Party Outfit' not 'Clothing'
        'Return Gifts' not 'Gifts'
        Think: decorations, food and drinks, tableware,
        party supplies, games, gifts, clothing, music setup.
        Consider the occasion and theme mentioned.
    """,

    "general": """
        You are an expert personal shopper and product specialist.
        List every item needed to completely fulfill the shopping goal.
        Use SPECIFIC product names as labels.
        NEVER use generic words like 'Items', 'Products',
        'Accessories', 'Supplies', 'Equipment'.
        Always be specific to what the query is asking for.
    """
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
        + "\n\nCRITICAL RULES:\n"
        + "1. Return ONLY a valid JSON object with a 'categories' key\n"
        + "2. Each category must have 'label', 'icon', 'search_terms'\n"
        + "3. Minimum 4 categories, maximum 10 categories\n"
        + "4. Labels must be SPECIFIC product names NEVER generic words\n"
        + "5. Search terms must be specific Amazon ready queries\n"
        + "6. Consider Indian market and pricing\n"
        + "7. Return ONLY JSON, no text, no markdown, no explanation\n"
    )

    human_content = (
        f"Shopping Goal: {query}\n"
        f"Domain: {domain}\n"
        f"Occasion: {occasion if occasion else 'not specified'}\n"
        f"Budget: {budget if budget else 'not specified'}\n"
        f"Persona: {persona if persona else 'not specified'}\n\n"
        "Return JSON with specific product categories and Amazon search terms."
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content)
    ]

    response = await llm.ainvoke(messages)
    result = parser.parse(response.content)

    print(f"RAW EXPANDER RESULT: {result}")
    return result