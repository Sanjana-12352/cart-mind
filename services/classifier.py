import os
import json
import time
import anthropic


SYSTEM_PROMPT = """Search intelligence for office/school supply platform.

CATALOG: Furniture, Technology, Office Supplies, Breakroom, Cleaning, School Supplies, Services, Shipping

9 INTENTS:
1. occasion_goal - broad goal, multiple categories
2. keyword_product - knows exactly what they want
3. brand_lookup - brand first
4. clarification_needed - ambiguous or missing info
5. b2b_escalation - qty>50, invoicing, business account
6. service_redirect - wants service done for them
7. replenishment - ran out, reordering
8. zero_result_rescue - not in catalog
9. attribute_question - asking about specs

RULES:
- 2+ categories → occasion_goal
- missing model/type → clarification_needed
- qty>50 → b2b_escalation
- service request → service_redirect
- out of catalog → zero_result_rescue
- replenishment + ambiguous → clarification_needed with options

OUTPUT: Return ONLY this JSON. Raw. No markdown.

For occasion_goal:
{"i":"occasion_goal","r":"multi_category_guided","c":0.95,"n":"normalized query","cats":[{"l":"Label","q":"br query"}],"msg":"headline|subtitle"}

For keyword_product:
{"i":"keyword_product","r":"direct_search","c":0.95,"n":"normalized","brand":null,"msg":"headline|subtitle"}

For brand_lookup:
{"i":"brand_lookup","r":"brand_filtered","c":0.95,"n":"normalized","brand":"brand name","msg":"headline|subtitle"}

For clarification_needed:
{"i":"clarification_needed","r":"clarification_needed","c":0.95,"n":"normalized","q":"question?","opts":[{"l":"label","v":"value"}],"msg":"headline|subtitle"}

For b2b_escalation:
{"i":"b2b_escalation","r":"b2b_escalation","c":0.95,"n":"normalized","reason":"short","msg":"headline|subtitle"}

For service_redirect:
{"i":"service_redirect","r":"service_redirect","c":0.95,"n":"normalized","svc":"print_services|tech_services|educator_discount|loyalty_program","msg":"headline|subtitle"}

For replenishment:
{"i":"replenishment","r":"replenishment","c":0.95,"n":"normalized","product":"short","msg":"headline|subtitle"}

For zero_result_rescue:
{"i":"zero_result_rescue","r":"zero_result_rescue","c":0.3,"n":"normalized","reason":"short","rescue":[{"l":"label","q":"br query"}],"msg":"headline|subtitle"}

For attribute_question:
{"i":"attribute_question","r":"attribute_question","c":0.95,"n":"normalized","attr":"attribute","msg":"headline|subtitle"}

RULES FOR COMPACT OUTPUT:
- cats: MAX 4 items
- opts: MAX 4 items
- rescue: MAX 3 items
- q in cats: MAX 4 words
- msg format: "headline|subtitle" as single pipe-separated string
- headline MAX 4 words
- subtitle MAX 8 words
- ALL values as short as possible
- TARGET: 150-200 tokens total"""


def expand_result(raw_result: dict, query: str) -> dict:
    """
    Expands the compact Claude response into full format
    so the API response is still rich and useful
    """
    intent = raw_result.get("i", "occasion_goal")
    routing = raw_result.get("r", "multi_category_guided")
    confidence = raw_result.get("c", 0.95)
    normalized = raw_result.get("n", query)

    msg_raw = raw_result.get("msg", "Results|Here are your results")
    msg_parts = msg_raw.split("|", 1)
    headline = msg_parts[0] if msg_parts else "Results"
    subtitle = msg_parts[1] if len(msg_parts) > 1 else "Find what you need."

    result = {
        "intent_class": intent,
        "routing": routing,
        "confidence": confidence,
        "normalized": normalized,
        "spell_corrected": normalized.lower() != query.lower(),
        "message": {
            "headline": headline,
            "subtitle": subtitle
        },
        "nav_categories": [],
        "clarification": None,
        "b2b_escalation": None,
        "service_redirect": None,
        "replenishment": None,
        "rescue_suggestions": [],
        "zero_result_rescue_triggered": intent == "zero_result_rescue",
        "out_of_catalog": intent == "zero_result_rescue",
        "brand_preference": raw_result.get("brand"),
        "educator_discount_prompt": False
    }

    if intent == "occasion_goal":
        cats = raw_result.get("cats", [])
        result["nav_categories"] = [
            {
                "label": c.get("l", ""),
                "category_id": f"cat_{c.get('l','x').lower().replace(' ','_')}",
                "plp_url": f"/category/{c.get('l','x').lower().replace(' ','-')}",
                "br_query": c.get("q", "")
            }
            for c in cats
        ]

    elif intent == "clarification_needed":
        result["clarification"] = {
            "question": raw_result.get("q", "Can you be more specific?"),
            "options": raw_result.get("opts", [])
        }

    elif intent == "b2b_escalation":
        result["b2b_escalation"] = {
            "reason": raw_result.get("reason", "Bulk order detected"),
            "action": "Contact our B2B team for pricing"
        }

    elif intent == "service_redirect":
        result["service_redirect"] = {
            "service_type": raw_result.get("svc", "print_services"),
            "cta": "Get started"
        }

    elif intent == "replenishment":
        result["replenishment"] = {
            "product_hint": raw_result.get("product", ""),
            "reorder_message": "Time to restock!"
        }

    elif intent == "zero_result_rescue":
        rescue = raw_result.get("rescue", [])
        result["rescue_suggestions"] = [
            {"label": r.get("l", ""), "br_query": r.get("q", "")}
            for r in rescue
        ]
        result["out_of_catalog_reason"] = raw_result.get("reason", "Not in catalog")

    elif intent == "attribute_question":
        result["attribute"] = raw_result.get("attr", "")

    return result


async def classify_query(query: str) -> dict:

    client = anthropic.AsyncAnthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    t_start = time.time()

    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=400,
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
            }
        ]
    )

    t_claude = time.time() - t_start

    if hasattr(response, "usage"):
        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0)
        cache_create = getattr(usage, "cache_creation_input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)

        if cache_read > 0:
            print(f" Cache HIT: {cache_read} tokens from cache")
        elif cache_create > 0:
            print(f" Cache CREATED: {cache_create} tokens cached")

        print(f" Claude API: {t_claude:.2f}s | Output tokens: {output_tokens}")

    raw = response.content[0].text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        compact_result = json.loads(raw)
        result = expand_result(compact_result, query)

    except json.JSONDecodeError as e:
        print(f" JSON error: {e}")
        result = {
            "intent_class": "occasion_goal",
            "routing": "multi_category_guided",
            "confidence": 0.7,
            "normalized": query,
            "spell_corrected": False,
            "message": {
                "headline": f"Shopping for {query}",
                "subtitle": "Find what you need."
            },
            "nav_categories": [],
            "clarification": None,
            "b2b_escalation": None,
            "service_redirect": None,
            "replenishment": None,
            "rescue_suggestions": [],
            "zero_result_rescue_triggered": False,
            "out_of_catalog": False
        }

    t_total = time.time() - t_start
    print(f" Total: {t_total:.2f}s | Intent: {result.get('intent_class')} | Routing: {result.get('routing')}")

    return result