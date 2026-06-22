from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from models.schemas import QueryRequest
from services.classifier import classify_query
import traceback

router = APIRouter()


@router.post("/v1/search/classify")
async def process_query(request: QueryRequest):

    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(query) > 500:
        raise HTTPException(status_code=400, detail="Query too long.")

    try:
        print(f"\n{'='*50}")
        print(f" Raw query: {query}")

        result = await classify_query(query)

        print(f"Response ready")
        print(f"{'='*50}\n")

        return JSONResponse(content=result)

    except HTTPException:
        raise

    except Exception as e:
        print(f"\n Pipeline error: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again."
        )