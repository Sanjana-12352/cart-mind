import os
key = os.getenv("OPENAI_API_KEY")
print(f"KEY LOADED: {key[:10] if key else 'NOT FOUND'}")
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.query import router as query_router
from routes.product import router as product_routes
from dotenv import load_dotenv
import os
load_dotenv()
app=FastAPI(
    title='CartMind API',
    description='Goal to Cart AI shopping Engine',
    version='1.0.0'
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://yourdomain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.include_router(
    query_router,
    prefix='/api',
    tags=["Query"]

)
app.include_router(
    product_routes,
    prefix="/api",
    tags=["Product"]
)
@app.get("/")
async def health_check():
    return {
        "status":"CartMind API is running",
        "version": "1.0.0"
    }
if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(
        "main:app",
        host='0.0.0.0',
        port=8000,
        reload=True
    )