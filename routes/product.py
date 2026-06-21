from fastapi import APIRouter, HTTPException
from models.schemas import ProductDetailResponse
from services.amazon import get_product_details, get_similar_products
import traceback

router = APIRouter()


@router.get("/product/{asin}", response_model=ProductDetailResponse)
async def get_product(asin: str):

    asin = asin.strip().upper()

    if not asin:
        raise HTTPException(status_code=400, detail="ASIN cannot be empty")

    if len(asin) != 10:
        raise HTTPException(status_code=400, detail="Invalid ASIN format. Must be 10 characters.")

    try:
        print(f"\n🔍 Fetching product details for ASIN: {asin}")

        product_details = await get_product_details(asin)

        if not product_details:
            raise HTTPException(status_code=404, detail=f"Product {asin} not found")

        print(f"✅ Product details fetched: {product_details.get('title', '')[:50]}")

        print(f"\n🔍 Fetching similar products...")

        similar_products = await get_similar_products(asin)

        print(f"✅ Found {len(similar_products)} similar products")

        response = {
            "asin"            : product_details.get("asin", asin),
            "title"           : product_details.get("title", ""),
            "description"     : product_details.get("description", ""),
            "price"           : product_details.get("price", 0.0),
            "rating"          : product_details.get("rating", 0.0),
            "total_reviews"   : product_details.get("total_reviews", 0),
            "images"          : product_details.get("images", []),
            "availability"    : product_details.get("availability", "In Stock"),
            "prime"           : product_details.get("prime", False),
            "brand"           : product_details.get("brand", ""),
            "category"        : product_details.get("category", ""),
            "affiliate_url"   : product_details.get("affiliate_url", ""),
            "similar_products": similar_products
        }

        print(f"✅ Response ready for product: {asin}")
        return response

    except HTTPException:
        raise

    except Exception as e:
        print(f"\n❌ Product fetch error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Something went wrong fetching product details. Please try again.")


@router.get("/product/{asin}/similar")
async def get_similar(asin: str):

    asin = asin.strip().upper()

    if not asin or len(asin) != 10:
        raise HTTPException(status_code=400, detail="Invalid ASIN")

    try:
        similar = await get_similar_products(asin)
        return {"similar_products": similar}

    except Exception as e:
        print(f"\n❌ Similar products error: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch similar products")