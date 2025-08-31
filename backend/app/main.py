from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import os

from .database import get_db, Generation, create_tables
from .models import GenerationRequest, GenerationResponse, GenerationHistory
from .shakespeare_model import ShakespeareModel

# Initialize FastAPI app
app = FastAPI(
    title="ShakespeareGPT API",
    description="API for generating Shakespeare-style text using a custom trained LLM",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
shakespeare_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global shakespeare_model
    
    # Get paths from environment or use defaults
    checkpoint_path = os.getenv("CHECKPOINT_PATH", "/app/checkpoint.pt")
    train_text_path = os.getenv("TRAIN_TEXT_PATH", "/app/train.txt")
    
    # Debug: Check if files exist
    print(f"🔍 Checking checkpoint path: {checkpoint_path}")
    print(f"🔍 File exists: {os.path.exists(checkpoint_path)}")
    if os.path.exists(checkpoint_path):
        print(f"🔍 File size: {os.path.getsize(checkpoint_path)} bytes")
    
    print(f"🔍 Checking train text path: {train_text_path}")
    print(f"🔍 File exists: {os.path.exists(train_text_path)}")
    if os.path.exists(train_text_path):
        print(f"🔍 File size: {os.path.getsize(train_text_path)} bytes")
    
    try:
        shakespeare_model = ShakespeareModel(checkpoint_path, train_text_path)
        print("✅ Shakespeare model loaded successfully!")
        
        # Create database tables
        create_tables()
        print("✅ Database tables created!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "ShakespeareGPT API is running!",
        "model_loaded": shakespeare_model is not None
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    db: Session = Depends(get_db)
):
    """Generate Shakespeare-style text from a prompt."""
    if shakespeare_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Generate text using the model
        generated_text = shakespeare_model.generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k
        )
        
        # Store in database
        db_generation = Generation(
            prompt=request.prompt,
            response=generated_text,
            temperature=request.temperature,
            top_k=request.top_k,
            max_new_tokens=request.max_new_tokens
        )
        
        db.add(db_generation)
        db.commit()
        db.refresh(db_generation)
        
        # Return response
        return GenerationResponse(
            id=db_generation.id,
            prompt=db_generation.prompt,
            response=db_generation.response,
            temperature=db_generation.temperature,
            top_k=db_generation.top_k,
            max_new_tokens=db_generation.max_new_tokens,
            created_at=db_generation.created_at
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/history", response_model=List[GenerationHistory])
async def get_generation_history(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get paginated history of text generations."""
    generations = db.query(Generation).order_by(
        Generation.created_at.desc()
    ).offset(offset).limit(limit).all()
    
    return [
        GenerationHistory(
            id=gen.id,
            prompt=gen.prompt,
            response=gen.response,
            temperature=gen.temperature,
            top_k=gen.top_k,
            max_new_tokens=gen.max_new_tokens,
            created_at=gen.created_at
        )
        for gen in generations
    ]

@app.get("/generation/{generation_id}", response_model=GenerationHistory)
async def get_generation_by_id(
    generation_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific generation by ID."""
    generation = db.query(Generation).filter(Generation.id == generation_id).first()
    
    if generation is None:
        raise HTTPException(status_code=404, detail="Generation not found")
    
    return GenerationHistory(
        id=generation.id,
        prompt=generation.prompt,
        response=generation.response,
        temperature=generation.temperature,
        top_k=generation.top_k,
        max_new_tokens=generation.max_new_tokens,
        created_at=generation.created_at
    )

@app.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get basic statistics about the API."""
    total_generations = db.query(Generation).count()
    
    return {
        "total_generations": total_generations,
        "model_loaded": shakespeare_model is not None,
        "model_device": shakespeare_model.device if shakespeare_model else None
    }
