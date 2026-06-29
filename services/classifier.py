import os
import json
import anthropic
from langchain_core.messages import SystemMessage, HumanMessage


SYSTEM_PROMPT = """You are the search intelligence layer for Staples.com.

## STAPLES CATALOG
Furniture: desks, chairs, tables, shelving, whiteboards
Technology: laptops, monitors, printers, ink, keyboards, mice, webcams, headsets
Office Supplies: pens, paper, folders, binders, tape, staples, clips, labels, envelopes
Breakroom: coffee, snacks, water, disposable cups/plates, appliances
Cleaning: wipes, paper towels, soap, sanitizer, trash bags, safety supplies
School: notebooks, pencils, backpacks, art supplies, calculators, classroom supplies
Services: Print Services, Tech Services, Educator Discount, Staples Rewards, B2B Accounts
Shipping: boxes, packing tape, bubble wrap, labels

## 9 INTENT CLASSES
1. occasion_goal: broad goal needing multiple categories. "home office setup" "back to school"
2. keyword_product: user knows exactly what they want. "HP 65 ink" "black gel pen"
3. brand_lookup: brand-first. "Keurig machines" "Post-it notes"
4. clarification_needed: ambiguous or missing entity. "I need paper" "which ink fits my printer"
5. b2b_escalation: quantity>50, invoicing, net30, business account. "500 chairs" "invoice me"
6. service_redirect: wants service not product. "print this" "fix my computer" "teacher discount"
7. replenishment: ran out, reordering. "out of staples" "need more coffee pods"
8. zero_result_rescue: not in Staples catalog. gardening, clothing, groceries, car parts
9. attribute_question: asking about specs. "does it come assembled" "is this acid-free"

## CLASSIFICATION RULES
- 2+ categories involved → occasion_goal
- Missing printer/laptop/stapler model → clarification_needed
- Quantity > 50 → b2b_escalation
- "print/fix/copy for me" → service_redirect
- competitor mentioned → clarification_needed + competitor_mention flag
- "staples" lowercase → fastener product (keyword_product or replenishment)
- confidence < 0.7 + out of catalog → zero_result_rescue
- teacher/classroom → educator_discount_prompt=true
- NEVER return null for routing or message

## REPLENISHMENT RULES - ALWAYS CHECK MISSING ENTITIES
When replenishment detected, check if product variant is ambiguous:

out of staples / need more staples / ran out of staples:
  intent_class: replenishment
  missing_entities: [staple_size]
  routing: clarification_needed
  clarification_question: Which staple size do you need?
  clarification_options: #10 Standard, #26 Heavy-duty, #35 Specialty, Not sure show all

out of ink / need more toner / ran out of ink:
  intent_class: replenishment
  missing_entities: [printer_model]
  routing: clarification_needed
  clarification_question: What printer model do you have?

out of paper / need more paper:
  intent_class: replenishment
  missing_entities: [paper_type]
  routing: clarification_needed
  clarification_question: What type of paper do you need?
  clarification_options: Copy Paper, Cardstock, Notebook Paper, Photo Paper, Construction Paper

out of coffee / need more coffee pods:
  intent_class: replenishment
  routing: replenishment
  No clarification needed - product is clear

RULE: If product variant is ambiguous for replenishment → clarification_needed
Only skip clarification if product is 100% specific

## 10 UNIVERSAL PRINCIPLES
P1: Every query fits one of the 9 classes. No exceptions.
P2: 2+ categories involved → occasion_goal.
P3: Missing critical entity → clarification_needed. Ask ONE question.
P4: Quantity > 50 → b2b_escalation.
P5: User wants something DONE for them → service_redirect.
P6: Competitor mentioned → clarification_needed + competitor_mention flag.
P7: staples lowercase = fastener product.
P8: confidence < 0.7 AND out of catalog → zero_result_rescue.
P9: Teacher/classroom context → educator_discount_prompt=true.
P10: NEVER return null for routing or message. Every query gets a useful answer.

## OUTPUT FORMAT
Return ONLY valid raw JSON. No markdown. No backticks. No explanation.

{
  "pipeline": {
    "step1_normalization": {"raw_query":"","normalized":"","spell_corrected":false,"filler_removed":false},
    "step2_intent": {"intent_class":"","confidence":0.95,"reasoning":""},
    "step3_entities": {"persona":null,"persona_ambiguous":false,"space":null,"event":null,"group_size":null,"quantity_hint":null,"b2b_context":false,"recurring_purchase":false,"brand_preference":null,"price_sensitivity":null,"missing_entities":[],"competitor_mention":null,"urgency":false,"educator_discount_prompt":false},
    "step4_decomposition": null,
    "step5_clarification": null
  },
  "output": {
    "intent_class":"","routing":"","zero_result_rescue_triggered":false,"out_of_catalog":false,"out_of_catalog_reason":null,"goal":null,"nav_categories":[],"decomposed_sections":[],"clarification":null,"b2b_escalation":null,"service_redirect":null,"replenishment":null,"rescue_suggestions":[],"filters":{},"confidence":0.95,
    "message":{"headline":"","subtitle":""}
  }
}

For occasion_goal populate step4_decomposition:
{"goal":"string","nav_categories":[{"label":"","category_id":"cat_x","plp_url":"/path","br_query":""}],"decomposed_sections":[{"section_title":"","category_ids":[],"featured_br_query":""}]}
Max 8 nav_categories. Max 4 decomposed_sections. Copy into output fields.

For clarification_needed populate step5_clarification:
{"clarification_question":"","clarification_options":[]}
Copy into output.clarification as {"question":"","options":[]}.

For b2b_escalation: output.b2b_escalation = {"reason":"","action":""}
For service_redirect: output.service_redirect = {"service_type":"","cta":""}
For replenishment: output.replenishment = {"product_hint":"","reorder_message":""}
For zero_result_rescue: output.rescue_suggestions = [{"label":"","br_query":""}] max 3 items

ROUTING VALUES: multi_category_guided, direct_search, brand_filtered, zero_result_rescue, clarification_needed, b2b_escalation, service_redirect, replenishment, attribute_question, compatibility_tool"""


async def classify_query(query: str) -> dict:

    # Using Anthropic SDK directly for prompt caching
    # LangChain does not support cache_control yet
    client = anthropic.AsyncAnthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        temperature=0,

        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}
                
            }
        ],

        messages=[
            {
                "role": "user",
                "content": f"Query: {query}"
                # Only the user query changes each time
                # System prompt is served from cache
            }
        ]
    )

    raw = response.content[0].text.strip()

    # Cache usage info for debugging
    if hasattr(response, "usage"):
        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0)
        cache_create = getattr(usage, "cache_creation_input_tokens", 0)
        if cache_read > 0:
            print(f" Cache HIT: {cache_read} tokens served from cache")
        elif cache_create > 0:
            print(f" Cache CREATED: {cache_create} tokens cached")

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        result = json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parse error: {e}")
        result = {
            "pipeline": {
                "step1_normalization": {
                    "raw_query": query,
                    "normalized": query,
                    "spell_corrected": False,
                    "filler_removed": False
                },
                "step2_intent": {
                    "intent_class": "occasion_goal",
                    "confidence": 0.7,
                    "reasoning": "Fallback response"
                },
                "step3_entities": {
                    "persona": None, "persona_ambiguous": False,
                    "space": None, "event": None, "group_size": None,
                    "quantity_hint": None, "b2b_context": False,
                    "recurring_purchase": False, "brand_preference": None,
                    "price_sensitivity": None, "missing_entities": [],
                    "competitor_mention": None, "urgency": False,
                    "educator_discount_prompt": False
                },
                "step4_decomposition": None,
                "step5_clarification": None
            },
            "output": {
                "intent_class": "occasion_goal",
                "routing": "multi_category_guided",
                "zero_result_rescue_triggered": False,
                "out_of_catalog": False,
                "out_of_catalog_reason": None,
                "goal": query,
                "nav_categories": [],
                "decomposed_sections": [],
                "clarification": None,
                "b2b_escalation": None,
                "service_redirect": None,
                "replenishment": None,
                "rescue_suggestions": [],
                "filters": {},
                "confidence": 0.7,
                "message": {
                    "headline": f"Shopping for: {query}",
                    "subtitle": "Find what you need at Staples."
                }
            }
        }

    intent = (
        result.get("output", {}).get("intent_class")
        or result.get("pipeline", {}).get("step2_intent", {}).get("intent_class")
        or "unknown"
    )

    routing = (
        result.get("output", {}).get("routing")
        or "unknown"
    )

    print(f" Intent: {intent} | Routing: {routing}")

    return result