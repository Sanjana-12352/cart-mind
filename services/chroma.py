import os
import json
import time
import chromadb
from chromadb.utils import embedding_functions
from typing import Optional, List

CACHE_EXPIRY_SECONDS = 7 * 24 * 60 * 60
SIMILARITY_THRESHOLD = 0.85
COLLECTION_NAME = "cartmind_search_cache"


def get_chroma_client():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client


def get_collection():
    client = get_chroma_client()

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


async def check_cache(search_term: str) -> Optional[List[dict]]:
    try:
        collection = get_collection()

        results = collection.query(
            query_texts=[search_term],
            n_results=1,
            include=["metadatas", "distances"]
        )

        if not results["ids"][0]:
            return None

        distance = results["distances"][0][0]
        similarity = 1 - distance

        if similarity < SIMILARITY_THRESHOLD:
            return None

        metadata = results["metadatas"][0][0]
        cached_at = metadata.get("cached_at", 0)
        age = time.time() - cached_at

        if age > CACHE_EXPIRY_SECONDS:
            return None

        products_json = metadata.get("products", "[]")
        products = json.loads(products_json)

        return products

    except Exception as e:
        print(f"Cache check error: {e}")
        return None


async def store_in_cache(search_term: str, products: List[dict]) -> None:
    try:
        collection = get_collection()

        cache_id = search_term.replace(" ", "_").lower()
        products_json = json.dumps(products)

        try:
            existing = collection.get(ids=[cache_id])
            if existing["ids"]:
                collection.update(
                    ids=[cache_id],
                    documents=[search_term],
                    metadatas=[{
                        "search_term": search_term,
                        "products": products_json,
                        "cached_at": time.time(),
                        "product_count": len(products)
                    }]
                )
                return
        except:
            pass

        collection.add(
            ids=[cache_id],
            documents=[search_term],
            metadatas=[{
                "search_term": search_term,
                "products": products_json,
                "cached_at": time.time(),
                "product_count": len(products)
            }]
        )

    except Exception as e:
        print(f"Cache store error: {e}")


async def clear_expired_cache() -> int:
    try:
        collection = get_collection()

        all_entries = collection.get(include=["metadatas"])

        if not all_entries["ids"]:
            return 0

        expired_ids = []
        current_time = time.time()

        for i, metadata in enumerate(all_entries["metadatas"]):
            cached_at = metadata.get("cached_at", 0)
            age = current_time - cached_at

            if age > CACHE_EXPIRY_SECONDS:
                expired_ids.append(all_entries["ids"][i])

        if expired_ids:
            collection.delete(ids=expired_ids)

        return len(expired_ids)

    except Exception as e:
        print(f"Cache clear error: {e}")
        return 0