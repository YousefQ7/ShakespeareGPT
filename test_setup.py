#!/usr/bin/env python3
"""
Test script to verify ShakespeareGPT setup
Run this after copying your model files to test everything works
"""

import os
import sys
import torch

def test_model_files():
    """Test if required model files exist"""
    print("🔍 Checking model files...")
    
    required_files = ['checkpoint.pt', 'train.txt']
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"✅ {file} found ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file} missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        print("Please copy these files from your ShakespeareLLM folder:")
        print("cp ../ShakespeareLLM/checkpoint.pt ./")
        print("cp ../ShakespeareLLM/train.txt ./")
        return False
    
    return True

def test_python_dependencies():
    """Test if Python dependencies can be imported"""
    print("\n🐍 Testing Python dependencies...")
    
    try:
        import fastapi
        print("✅ FastAPI imported successfully")
    except ImportError:
        print("❌ FastAPI not found - run: pip install -r backend/requirements.txt")
        return False
    
    try:
        import sqlalchemy
        print("✅ SQLAlchemy imported successfully")
    except ImportError:
        print("❌ SQLAlchemy not found - run: pip install -r backend/requirements.txt")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch imported successfully (version {torch.__version__})")
    except ImportError:
        print("❌ PyTorch not found - run: pip install -r backend/requirements.txt")
        return False
    
    return True

def test_model_loading():
    """Test if the Shakespeare model can be loaded"""
    print("\n🤖 Testing model loading...")
    
    try:
        # Add backend to path
        sys.path.append('backend')
        
        from app.shakespeare_model import ShakespeareModel
        
        print("✅ ShakespeareModel class imported successfully")
        
        # Try to load the model
        model = ShakespeareModel('checkpoint.pt', 'train.txt')
        print("✅ Model loaded successfully!")
        print(f"   Device: {model.device}")
        print(f"   Vocabulary size: {model.vocab_size}")
        
        # Test a simple generation
        test_prompt = "Hello, world"
        print(f"\n🧪 Testing generation with prompt: '{test_prompt}'")
        
        generated = model.generate_text(
            prompt=test_prompt,
            max_new_tokens=20,
            temperature=1.0
        )
        
        print(f"✅ Generation successful!")
        print(f"   Generated: {generated}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_database_connection():
    """Test database connection (SQLite)"""
    print("\n🗄️ Testing database connection...")
    
    try:
        # Set SQLite database URL for testing
        os.environ['DATABASE_URL'] = 'sqlite:///./test.db'
        
        from backend.app.database import engine, create_tables
        
        # Test creating tables
        create_tables()
        print("✅ Database modules imported successfully")
        print("✅ SQLite database tables created successfully")
        
        # Clean up test database
        if os.path.exists('test.db'):
            os.remove('test.db')
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎭 ShakespeareGPT Setup Test")
    print("=" * 40)
    
    tests = [
        ("Model Files", test_model_files),
        ("Python Dependencies", test_python_dependencies),
        ("Model Loading", test_model_loading),
        ("Database Setup", test_database_connection),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ShakespeareGPT is ready to run.")
        print("\n🚀 To start the application:")
        print("   ./start.sh")
        print("\n📚 Or manually:")
        print("   cd backend && uvicorn app.main:app --reload")
        print("   cd frontend && npm run dev")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\n💡 Common solutions:")
        print("   1. Copy model files: cp ../ShakespeareLLM/checkpoint.pt ./")
        print("   2. Install dependencies: pip install -r backend/requirements.txt")
        print("   3. Check file paths and permissions")

if __name__ == "__main__":
    main()
