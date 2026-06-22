
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.query import router as query_router
from dotenv import load_dotenv
import os
load_dotenv()
app=FastAPI(
    title='CartMind API',
    description='Goal to Cart AI shopping Engine',
    version='2.0.0'
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(query_router, prefix="/api", tags=["Search"])
@app.get("/")
async def health_check():
    return {
        "status": "CartMind API is running",
        "version": "2.0.0",
        "powered_by": "Claude Sonnet 4.6"
    }
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
 