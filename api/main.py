import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure the root directory is in the path so 'api.xxx' imports work everywhere
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.database import connect_db, close_db
from api.routes.auth_routes import router as auth_router
from api.routes.user_routes import router as user_router
from api.routes.detection_routes import router as detection_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    yield
    close_db()

app = FastAPI(
    title="ShopGuard API",
    description="Backend API for Shoplifting Detection System",
    version="1.0.0",
    lifespan=lifespan,
)

# Robust CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev to solve the CORS block
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(auth_router)
app.include_router(user_router)
app.include_router(detection_router)

@app.get("/")
async def root():
    return {"message": "ShopGuard API is running"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
