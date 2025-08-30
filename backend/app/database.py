from sqlalchemy import create_engine, Column, Integer, Text, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration - SQLite for local dev, PostgreSQL for production
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./shakespearegpt.db")

# Create engine with appropriate settings
if "sqlite" in DATABASE_URL:
    # SQLite for local development
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    print("🗄️ Using SQLite database for local development")
else:
    # PostgreSQL for production
    engine = create_engine(DATABASE_URL)
    print("🗄️ Using PostgreSQL database for production")

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Database model
class Generation(Base):
    __tablename__ = "generations"
    
    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    temperature = Column(Float)
    top_k = Column(Integer)
    max_new_tokens = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
