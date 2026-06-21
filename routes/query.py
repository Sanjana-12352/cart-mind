from fastapi import APIRouter, HTTPException
from models.schemas import QueryRequest, QueryResponse
from services.intent import classify_intent
from services.expander import expand_query
from services.budget import distribute_budget
from services.optimizer import optimize_search_terms, get_terms_for_category
from services.amazon import search_all_categories
import asyncio
import traceback

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):

    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(query) < 3:
        raise HTTPException(status_code=400, detail="Query too short. Please be more specific.")
    if len(query) > 500:
        raise HTTPException(status_code=400, detail="Query too long. Please keep it under 500 characters.")

    try:
        print(f"\n🔍 Processing query: {query}")

        # ── STAGE 1: CLASSIFY INTENT ──
        # Must run first — all other stages depend on it
        intent_result = await classify_intent(query)
        print(f"✅ Intent classified: {intent_result}")

        domain   = intent_result.get("domain", "general")
        occasion = intent_result.get("occasion")
        budget   = intent_result.get("budget")
        currency = intent_result.get("currency", "INR")
        persona  = intent_result.get("persona")

        # ── STAGE 2: RUN EXPAND IN PARALLEL ──
        # expand_query runs first to get category labels
        # budget and optimize need category labels
        # so expand must finish before budget + optimize start
        print(f"\n🔍 Expanding query for domain: {domain}")

        expanded_result = await expand_query(
            query    = query,
            domain   = domain,
            occasion = occasion,
            budget   = budget,
            persona  = persona
        )

        expanded_categories = expanded_result.get("categories", [])
        print(f"✅ Expanded into {len(expanded_categories)} categories")

        if not expanded_categories:
            raise HTTPException(status_code=500, detail="Could not expand query into categories")

        category_labels = [cat.get("label", "") for cat in expanded_categories]

        # ── STAGE 3: RUN BUDGET + OPTIMIZE IN PARALLEL ──
        # Both only need category_labels and intent
        # Run them simultaneously to save time
        print(f"\n⚡ Running budget + optimize in parallel...")

        async def run_budget():
            if budget:
                return await distribute_budget(
                    budget     = budget,
                    domain     = domain,
                    categories = category_labels
                )
            return None

        async def run_optimize():
            return await optimize_search_terms(
                query      = query,
                domain     = domain,
                categories = category_labels,
                occasion   = occasion,
                persona    = persona
            )

        # Run both at the same time
        budget_result, optimized_result = await asyncio.gather(
            run_budget(),
            run_optimize()
        )

        budget_split = budget_result.get("allocations", {}) if budget_result else None
        print(f"✅ Budget + Optimize done")

        # ── STAGE 4: BUILD CATEGORIES WITH TERMS ──
        categories_with_terms = []
        for category in expanded_categories:
            label = category.get("label", "")
            icon  = category.get("icon", "🛍️")
            terms = get_terms_for_category(optimized_result, label)
            categories_with_terms.append({
                "label"          : label,
                "icon"           : icon,
                "optimized_terms": terms
            })

        # ── STAGE 5: FETCH AMAZON PRODUCTS ──
        print(f"\n🛒 Fetching Amazon products in parallel...")

        categories_with_products = await search_all_categories(
            categories   = categories_with_terms,
            budget_split = budget_split
        )
        print(f"✅ Products fetched")

        # ── ASSEMBLE RESPONSE ──
        labels = [cat["label"] for cat in categories_with_products]

        intent_obj = {
            "domain"  : domain,
            "occasion": occasion,
            "budget"  : budget,
            "currency": currency,
            "persona" : persona
        }

        response = {
            "intent"      : intent_obj,
            "labels"      : labels,
            "budget_split": budget_split,
            "categories"  : categories_with_products
        }

        print(f"\n✅ Response ready with {len(labels)} categories")
        return response

    except HTTPException:
        raise

    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Something went wrong processing your query. Please try again.")