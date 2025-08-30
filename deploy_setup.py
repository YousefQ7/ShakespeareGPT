#!/usr/bin/env python3
"""
Deployment Setup Script for ShakespeareGPT
This script helps verify your deployment configuration and provides helpful information.
"""

import os
import sys
from pathlib import Path

def check_files():
    """Check if all required files exist."""
    print("🔍 Checking required files...")
    
    required_files = [
        "backend/checkpoint.pt",
        "backend/train.txt", 
        "backend/requirements.txt",
        "backend/app/main.py",
        "backend/app/database.py",
        "backend/app/models.py",
        "backend/app/shakespeare_model.py",
        "frontend/package.json",
        "frontend/src/App.jsx"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files found!")
    return True

def check_model_size():
    """Check model file sizes."""
    print("\n📊 Checking model file sizes...")
    
    checkpoint_size = Path("backend/checkpoint.pt").stat().st_size / (1024 * 1024)  # MB
    train_size = Path("backend/train.txt").stat().st_size / (1024 * 1024)  # MB
    
    print(f"📁 checkpoint.pt: {checkpoint_size:.1f} MB")
    print(f"📁 train.txt: {train_size:.1f} MB")
    
    if checkpoint_size > 500:
        print("⚠️  Warning: Model file is large (>500MB). Consider using model compression.")
    
    return True

def print_deployment_info():
    """Print deployment information."""
    print("\n🚀 Deployment Information:")
    print("=" * 50)
    print("📋 Backend (Render):")
    print("   - Root Directory: backend/")
    print("   - Build Command: pip install -r requirements.txt")
    print("   - Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT")
    print("   - Environment Variables:")
    print("     * DATABASE_URL (from Supabase)")
    print("     * CHECKPOINT_PATH: /opt/render/project/src/checkpoint.pt")
    print("     * TRAIN_TEXT_PATH: /opt/render/project/src/train.txt")
    
    print("\n📋 Frontend (Vercel):")
    print("   - Root Directory: frontend/")
    print("   - Build Command: npm run build")
    print("   - Environment Variables:")
    print("     * VITE_API_URL (your Render backend URL)")
    
    print("\n📋 Database (Supabase):")
    print("   - Free tier: 500MB storage")
    print("   - Current usage: ~24KB (very small!)")
    print("   - Perfect for your project!")

def main():
    """Main function."""
    print("🎭 ShakespeareGPT Deployment Setup")
    print("=" * 40)
    
    # Check files
    if not check_files():
        print("\n❌ Please fix missing files before deploying.")
        sys.exit(1)
    
    # Check model size
    check_model_size()
    
    # Print deployment info
    print_deployment_info()
    
    print("\n✅ Deployment setup complete!")
    print("\n📝 Next steps:")
    print("1. Create Supabase account and get DATABASE_URL")
    print("2. Deploy backend to Render")
    print("3. Deploy frontend to Vercel")
    print("4. Test your live application!")

if __name__ == "__main__":
    main()
