# 🚀 Deployment Guide

This guide will walk you through deploying ShakespeareGPT to production with minimal cost.

## 🎯 Deployment Options

### Option 1: Railway (Recommended - Lowest Cost)
- **Backend**: $5/month for 512MB RAM
- **Database**: $5/month for PostgreSQL
- **Total**: ~$10/month

### Option 2: Render
- **Backend**: Free tier (sleeps after 15 min inactivity)
- **Database**: $7/month for PostgreSQL
- **Total**: ~$7/month

## 🚀 Railway Deployment

### Step 1: Prepare Your Repository

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: ShakespeareGPT full-stack app"
   git branch -M main
   git remote add origin https://github.com/yourusername/ShakespeareGPT.git
   git push -u origin main
   ```

2. **Ensure these files are in your repo**:
   - `backend/` folder with all Python files
   - `checkpoint.pt` (your trained model)
   - `train.txt` (training dataset)

### Step 2: Deploy Backend

1. **Go to [Railway.app](https://railway.app)** and sign in with GitHub
2. **Create New Project** → **Deploy from GitHub repo**
3. **Select your ShakespeareGPT repository**
4. **Set Root Directory** to `backend/`
5. **Add Environment Variables**:
   ```
   DATABASE_URL=postgresql://... (Railway will provide this)
   CHECKPOINT_PATH=/app/checkpoint.pt
   TRAIN_TEXT_PATH=/app/train.txt
   ```
6. **Deploy** and wait for build to complete

### Step 3: Set Up Database

1. **Add PostgreSQL** to your Railway project
2. **Copy the DATABASE_URL** from the PostgreSQL service
3. **Update your backend environment variables** with the new DATABASE_URL
4. **Redeploy** the backend

### Step 4: Deploy Frontend

1. **Create a new Railway project** for the frontend
2. **Set Root Directory** to `frontend/`
3. **Add Build Command**: `npm run build`
4. **Add Start Command**: `npx serve -s dist -l $PORT`
5. **Add Environment Variables**:
   ```
   VITE_API_URL=https://your-backend-url.railway.app
   ```

## 🌐 Render Deployment

### Step 1: Deploy Backend

1. **Go to [Render.com](https://render.com)** and sign in
2. **Create New** → **Web Service**
3. **Connect your GitHub repository**
4. **Configure**:
   - **Name**: `shakespearegpt-backend`
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. **Add Environment Variables**:
   ```
   CHECKPOINT_PATH=/opt/render/project/src/checkpoint.pt
   TRAIN_TEXT_PATH=/opt/render/project/src/train.txt
   ```

### Step 2: Set Up Database

1. **Create New** → **PostgreSQL**
2. **Copy the DATABASE_URL**
3. **Add to your backend environment variables**
4. **Redeploy** the backend

### Step 3: Deploy Frontend

1. **Create New** → **Static Site**
2. **Connect your GitHub repository**
3. **Configure**:
   - **Root Directory**: `frontend`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `dist`

## 🔧 Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host:port/db` |
| `CHECKPOINT_PATH` | Path to model checkpoint | `/app/checkpoint.pt` |
| `TRAIN_TEXT_PATH` | Path to training data | `/app/train.txt` |
| `PORT` | Server port (Railway/Render sets this) | `8000` |

## 📁 File Structure for Deployment

```
ShakespeareGPT/
├── backend/                 # Backend deployment root
│   ├── app/                # FastAPI application
│   ├── requirements.txt    # Python dependencies
│   ├── Dockerfile         # Container config
│   └── checkpoint.pt      # Model weights (copy from ShakespeareLLM/)
├── frontend/               # Frontend deployment root
│   ├── src/               # React source code
│   ├── package.json       # Node.js dependencies
│   └── train.txt          # Training data (copy from ShakespeareLLM/)
└── README.md              # Documentation
```

## 🐳 Docker Deployment (Alternative)

If you prefer Docker:

```bash
# Build backend image
cd backend
docker build -t shakespearegpt-backend .

# Run with environment variables
docker run -p 8000:8000 \
  -e DATABASE_URL="your_postgres_url" \
  -e CHECKPOINT_PATH="/app/checkpoint.pt" \
  -e TRAIN_TEXT_PATH="/app/train.txt" \
  shakespearegpt-backend
```

## 🔍 Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure `checkpoint.pt` and `train.txt` are in the correct paths
   - Check file permissions and sizes

2. **Database Connection Errors**:
   - Verify `DATABASE_URL` is correct
   - Ensure PostgreSQL service is running
   - Check firewall settings

3. **Build Failures**:
   - Verify all dependencies are in `requirements.txt`
   - Check Python version compatibility
   - Ensure all files are committed to Git

### Debug Commands

```bash
# Check backend logs
railway logs

# Check database connection
railway run psql $DATABASE_URL

# Test model loading locally
cd backend
python -c "from app.shakespeare_model import ShakespeareModel; m = ShakespeareModel('../checkpoint.pt', '../train.txt')"
```

## 📊 Monitoring & Scaling

### Railway Monitoring
- **Logs**: Available in Railway dashboard
- **Metrics**: CPU, memory, and request counts
- **Scaling**: Auto-scaling based on demand

### Render Monitoring
- **Logs**: Available in Render dashboard
- **Metrics**: Basic performance metrics
- **Scaling**: Manual scaling options

## 💰 Cost Optimization

### Railway
- **Development**: Use free tier for testing
- **Production**: Start with $5/month plan
- **Scale up**: Only when needed

### Render
- **Free tier**: Good for development
- **Paid plans**: Start with $7/month for database
- **Sleep mode**: Free tier sleeps after inactivity

## 🎯 Next Steps

After successful deployment:

1. **Test all endpoints** using the API docs
2. **Monitor performance** and logs
3. **Set up custom domain** (optional)
4. **Add SSL certificates** (automatic on Railway/Render)
5. **Implement monitoring** and alerting
6. **Add user authentication** for production use

---

**Happy deploying! 🚀✨**
