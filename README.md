# 🎭 ShakespeareGPT

A full-stack web application that generates Shakespeare-style text using a custom-trained language model. Built with FastAPI, React, and PostgreSQL.

## ✨ Features

- **AI Text Generation**: Generate Shakespeare-style text from custom prompts
- **Advanced Controls**: Adjust temperature, top-k, and max tokens for fine-tuned generation
- **Generation History**: View and search through all past generations
- **Modern UI**: Beautiful, responsive interface built with React and Tailwind CSS
- **RESTful API**: Clean FastAPI backend with comprehensive endpoints
- **Database Storage**: PostgreSQL database to persist all generations

## 🏗️ Architecture

```
ShakespeareGPT/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── main.py         # FastAPI application
│   │   ├── models.py       # Pydantic models
│   │   ├── database.py     # Database configuration
│   │   └── shakespeare_model.py  # Custom LLM wrapper
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile         # Container configuration
├── frontend/               # React frontend
│   ├── src/
│   │   ├── App.jsx        # Main application
│   │   ├── components/    # React components
│   │   └── index.css      # Tailwind CSS styles
│   ├── package.json       # Node.js dependencies
│   └── tailwind.config.js # Tailwind configuration
├── checkpoint.pt           # Trained model weights
├── train.txt              # Training dataset (for vocabulary)
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- PostgreSQL database
- Your trained model files from `ShakespeareLLM/` folder

### 1. Copy Model Files

First, copy these essential files from your `ShakespeareLLM/` folder:

```bash
cp ../ShakespeareLLM/checkpoint.pt ./
cp ../ShakespeareLLM/train.txt ./
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://user:password@localhost/shakespearegpt"
export CHECKPOINT_PATH="../checkpoint.pt"
export TRAIN_TEXT_PATH="../train.txt"

# Run the backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000` and will proxy API calls to the backend at `http://localhost:8000`.

## 🌐 API Endpoints

### Text Generation
- `POST /generate` - Generate Shakespeare-style text
  - Body: `{prompt, temperature?, top_k?, max_new_tokens?}`
  - Returns: `{id, prompt, response, temperature, top_k, max_new_tokens, created_at}`

### History & Retrieval
- `GET /history?limit=20&offset=0` - Get paginated generation history
- `GET /generation/{id}` - Get specific generation by ID
- `GET /stats` - Get API statistics

### Health Check
- `GET /` - API health and model status

## 🗄️ Database Schema

```sql
CREATE TABLE generations (
    id SERIAL PRIMARY KEY,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    temperature REAL,
    top_k INT,
    max_new_tokens INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🐳 Docker Deployment

### Backend Container

```bash
cd backend
docker build -t shakespearegpt-backend .
docker run -p 8000:8000 \
  -e DATABASE_URL="your_postgres_url" \
  -e CHECKPOINT_PATH="/app/checkpoint.pt" \
  -e TRAIN_TEXT_PATH="/app/train.txt" \
  shakespearegpt-backend
```

### Frontend Container

```bash
cd frontend
npm run build
# Serve the dist/ folder with nginx or similar
```

## 🚀 Production Deployment

### Railway (Recommended for Low Cost)

1. **Backend Deployment**:
   - Connect your GitHub repo to Railway
   - Set environment variables:
     - `DATABASE_URL`: Railway PostgreSQL URL
     - `CHECKPOINT_PATH`: `/app/checkpoint.pt`
     - `TRAIN_TEXT_PATH`: `/app/train.txt`
   - Deploy from the `backend/` directory

2. **Database Setup**:
   - Railway automatically creates PostgreSQL
   - Tables are created automatically on startup

3. **Frontend Deployment**:
   - Build and deploy to Vercel, Netlify, or Railway
   - Update API base URL to your Railway backend URL

### Render Alternative

- Similar process to Railway
- Use Render's PostgreSQL service
- Deploy backend as a Web Service

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:password@localhost/shakespearegpt` |
| `CHECKPOINT_PATH` | Path to model checkpoint | `../checkpoint.pt` |
| `TRAIN_TEXT_PATH` | Path to training data | `../train.txt` |

### Model Parameters

The model supports these generation parameters:

- **Temperature** (0.1 - 2.0): Controls randomness (lower = more focused)
- **Top-K** (1 - 100): Limits token selection to top K most likely
- **Max New Tokens** (10 - 500): Maximum length of generated text

## 🧪 Testing

### Backend Testing

```bash
cd backend
pytest  # If you add tests
```

### Frontend Testing

```bash
cd frontend
npm test  # If you add tests
```

## 📊 Performance Considerations

- **Model Loading**: The model loads on startup and stays in memory
- **Generation Speed**: Depends on your hardware (CPU/GPU)
- **Database**: PostgreSQL handles concurrent requests efficiently
- **Caching**: Consider adding Redis for frequently accessed data

## 🔒 Security Notes

- CORS is currently set to allow all origins (customize for production)
- No authentication system (add JWT tokens for production)
- Input validation with Pydantic models
- SQL injection protection via SQLAlchemy ORM

## 🚧 Future Enhancements

- [ ] User authentication and accounts
- [ ] Rate limiting
- [ ] Model fine-tuning interface
- [ ] Export generations to various formats
- [ ] Real-time generation streaming
- [ ] Multiple model support
- [ ] API key management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with your custom-trained ShakespeareLLM
- Powered by PyTorch and FastAPI
- Beautiful UI with React and Tailwind CSS
- Database powered by PostgreSQL

---

**Happy generating! 🎭✨**
