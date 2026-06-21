import os
import httpx
import asyncio
from typing import Optional, List
from services.chroma import check_cache, store_in_cache

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")
PARTNER_TAG = os.getenv("AMAZON_PARTNER_TAG")

BASE_URL = "https://real-time-amazon-data.p.rapidapi.com"

HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST,
    "Content-Type": "application/json"
}

PRODUCTS_PER_CATEGORY = 10


def build_affiliate_url(asin: str) -> str:
    return f"https://www.amazon.in/dp/{asin}?tag={PARTNER_TAG}"


def parse_price(price_str: str) -> float:
    try:
        cleaned = price_str.replace("₹", "") \
                           .replace("$", "") \
                           .replace(",", "") \
                           .strip()
        return float(cleaned)
    except:
        return 0.0


def parse_product(raw_product: dict) -> Optional[dict]:
    try:
        asin = raw_product.get("asin")
        if not asin:
            return None

        title = raw_product.get("product_title", "")
        if not title:
            return None

        price_str = raw_product.get("product_price", "0")
        price = parse_price(price_str)
        if price <= 0:
            return None

        rating_str = raw_product.get("product_star_rating", "0")
        try:
            rating = float(rating_str)
        except:
            rating = 0.0

        reviews_str = raw_product.get("product_num_ratings", "0")
        try:
            total_reviews = int(str(reviews_str).replace(",", ""))
        except:
            total_reviews = 0

        image = raw_product.get("product_photo", "")
        if not image:
            image = raw_product.get("product_main_image_url", "")

        prime = raw_product.get("is_prime", False)

        availability = raw_product.get("product_availability", "In Stock")
        in_stock = "stock" in availability.lower() or \
                   "available" in availability.lower()

        affiliate_url = build_affiliate_url(asin)

        return {
            "asin": asin,
            "title": title,
            "price": price,
            "rating": rating,
            "total_reviews": total_reviews,
            "image": image,
            "affiliate_url": affiliate_url,
            "prime": prime,
            "in_stock": in_stock
        }

    except Exception as e:
        print(f"Product parse error: {e}")
        return None


async def search_products(
    search_term: str,
    budget_cap: Optional[float] = None,
    country: str = "IN"
) -> List[dict]:

    cached = await check_cache(search_term)
    if cached:
        if budget_cap:
            cached = [p for p in cached if p["price"] <= budget_cap]
        return cached[:PRODUCTS_PER_CATEGORY]

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{BASE_URL}/search",
                headers=HEADERS,
                params={
                    "query": search_term,
                    "page": "1",
                    "country": country,
                    "sort_by": "RELEVANCE",
                    "product_condition": "NEW",
                    "is_prime": "false"
                }
            )

            if response.status_code != 200:
                print(f"RapidAPI error: {response.status_code}")
                return []

            data = response.json()
            raw_products = data.get("data", {}).get("products", [])

            if not raw_products:
                return []

            parsed_products = []
            for raw in raw_products:
                parsed = parse_product(raw)
                if parsed:
                    parsed_products.append(parsed)

            if budget_cap:
                parsed_products = [
                    p for p in parsed_products
                    if p["price"] <= budget_cap
                ]

            parsed_products.sort(
                key=lambda p: (p["rating"], p["total_reviews"]),
                reverse=True
            )

            final_products = parsed_products[:PRODUCTS_PER_CATEGORY]

            if final_products:
                await store_in_cache(search_term, final_products)

            return final_products

    except httpx.TimeoutException:
        print(f"RapidAPI timeout for: {search_term}")
        return []
    except Exception as e:
        print(f"Amazon search error: {e}")
        return []


async def search_all_categories(
    categories: List[dict],
    budget_split: Optional[dict] = None
) -> List[dict]:

    async def search_one_category(category: dict) -> dict:
        label = category.get("label", "")
        icon = category.get("icon", "🛍️")
        terms = category.get("optimized_terms", [])

        budget_cap = None
        if budget_split:
            budget_cap = budget_split.get(label)

        # Use ONLY the first (best) search term per category
        # This reduces API calls from 18 to 6
        # dramatically improving response time
        best_term = terms[0] if terms else label
        products = await search_products(best_term, budget_cap)

        return {
            "label": label,
            "icon": icon,
            "products": products
        }

    tasks = [search_one_category(cat) for cat in categories]
    results = await asyncio.gather(*tasks)

    return list(results)


async def get_product_details(asin: str) -> Optional[dict]:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{BASE_URL}/product-details",
                headers=HEADERS,
                params={
                    "asin": asin,
                    "country": "IN"
                }
            )

            if response.status_code != 200:
                return None

            data = response.json()
            product_data = data.get("data", {})

            if not product_data:
                return None

            asin_val = product_data.get("asin", asin)
            title = product_data.get("product_title", "")
            price_str = product_data.get("product_price", "0")
            price = parse_price(price_str)

            rating_str = product_data.get("product_star_rating", "0")
            try:
                rating = float(rating_str)
            except:
                rating = 0.0

            reviews_str = product_data.get("product_num_ratings", "0")
            try:
                total_reviews = int(str(reviews_str).replace(",", ""))
            except:
                total_reviews = 0

            images = product_data.get("product_photos", [])
            if not images:
                main_image = product_data.get("product_main_image_url")
                if main_image:
                    images = [main_image]

            description = product_data.get("product_description", "")
            brand = product_data.get("brand", "")
            availability = product_data.get("product_availability", "In Stock")
            prime = product_data.get("is_prime", False)
            affiliate_url = build_affiliate_url(asin_val)

            return {
                "asin": asin_val,
                "title": title,
                "description": description,
                "price": price,
                "rating": rating,
                "total_reviews": total_reviews,
                "images": images,
                "availability": availability,
                "prime": prime,
                "brand": brand,
                "affiliate_url": affiliate_url,
            }

    except Exception as e:
        print(f"Product details error: {e}")
        return None


async def get_similar_products(asin: str) -> List[dict]:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{BASE_URL}/product-details",
                headers=HEADERS,
                params={
                    "asin": asin,
                    "country": "IN"
                }
            )

            if response.status_code != 200:
                return []

            data = response.json()
            product_data = data.get("data", {})
            similar_raw = product_data.get("similar_products", [])

            similar = []
            for raw in similar_raw[:5]:
                parsed = parse_product(raw)
                if parsed:
                    similar.append(parsed)

            return similar

    except Exception as e:
        print(f"Similar products error: {e}")
        return []