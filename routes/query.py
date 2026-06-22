from fastapi import APIRouter, HTTPException
from models.schemas import QueryRequest, QueryResponse
from services.normalize import normalize_query
from services.intent import classify_intent
from services.expander import expand_query
from services.product_names import generate_product_names
import traceback

router = APIRouter()


@router.post("/v1/search/classify", response_model=QueryResponse)
async def process_query(request: QueryRequest):

    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(query) > 500:
        raise HTTPException(status_code=400, detail="Query too long. Please keep it under 500 characters.")

    try:
        print(f"\n{'='*50}")
        print(f" Raw query: {query}")

        normalized = await normalize_query(query)
        intent_result = await classify_intent(normalized)

        domain   = intent_result.get("domain", "general")
        occasion = intent_result.get("occasion")
        budget   = intent_result.get("budget")
        currency = intent_result.get("currency", "INR")
        persona  = intent_result.get("persona")
        quantity = intent_result.get("quantity")
        print(f"\n Expanding for domain: {domain}")

        expanded_result = await expand_query(
            query    = normalized,
            domain   = domain,
            occasion = occasion,
            persona  = persona,
            quantity = quantity
        )

        categories = expanded_result.get("categories", [])

        if not categories:
            raise HTTPException(
                status_code=500,
                detail="Could not expand query into categories"
            )

        # ── STAGE 4: GENERATE PRODUCT NAMES ──
        print(f"\n Generating product names...")

        categories_with_products = await generate_product_names(
            query      = normalized,
            categories = categories,
            domain     = domain,
            occasion   = occasion,
            quantity   = quantity
        )

        # ── ASSEMBLE RESPONSE ──
        labels = [cat["label"] for cat in categories_with_products]

        intent_obj = {
            "domain"  : domain,
            "occasion": occasion,
            "budget"  : budget,
            "currency": currency,
            "persona" : persona,
            "quantity": quantity
        }

        response = {
            "query"     : normalized,
            "intent"    : intent_obj,
            "labels"    : labels,
            "categories": categories_with_products
        }

        print(f"\n Response ready: {len(labels)} categories")
        print(f"{'='*50}\n")

        return response

    except HTTPException:
        raise

    except Exception as e:
        print(f"\n Pipeline error: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again."
        )