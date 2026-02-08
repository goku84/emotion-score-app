# Deploying the Fake Review Detection System

This guide outlines how to deploy both the Python FastAPI backend and the Next.js frontend to the cloud for free using popular services.

---

## 🚀 Option 1: Render (Backend) + Vercel (Frontend) [Recommended]

This is the easiest way to deploy each part where it runs best.

### Part A: Deploying the Backend (Python/FastAPI) on Render

1.  **Push your code to GitHub**: Make sure your project is in a GitHub repository.
2.  **Sign up/Log in to Render** (https://render.com).
3.  Click **New +** and select **Web Service**.
4.  Connect your GitHub repository.
5.  **Configure the Service**:
    *   **Name**: `fake-review-backend` (or similar)
    *   **Root Directory**: `backend` (Important: tell Render the Python app is in this folder)
    *   **Environment**: `Python 3`
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6.  Click **Create Web Service**.
7.  Wait for the deployment to finish. Once done, copy the **URL** (e.g., `https://fake-review-backend.onrender.com`).

**Note**: The first build might take a few minutes as it downloads large ML models (PyTorch, Spacy, Transformers).

### Part B: Deploying the Frontend (Next.js) on Vercel

1.  **Sign up/Log in to Vercel** (https://vercel.com).
2.  Click **Add New...** -> **Project**.
3.  Import your GitHub repository.
4.  **Configure the Project**:
    *   **Framework Preset**: Next.js (should be auto-detected).
    *   **Root Directory**: Click "Edit" and select the `frontend` folder.
5.  **Environment Variables**:
    *   Expand the "Environment Variables" section.
    *   Key: `NEXT_PUBLIC_API_URL`
    *   Value: Your backend URL from Part A (e.g., `https://fake-review-backend.onrender.com`). **Do not add a trailing slash**.
6.  Click **Deploy**.
7.  Vercel will build your frontend. Once complete, you will get a live URL (e.g., `https://fake-review-frontend.vercel.app`).

---

## 🐳 Option 2: Docker / VPS (Advanced)

If you prefer to deploy on a single server (like DigitalOcean, AWS EC2, or locally), you can use Docker.

### 1. Create a `Dockerfile` for Backend

Create `backend/Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
# (This includes the spacy model url in requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Create a `Dockerfile` for Frontend

Create `frontend/Dockerfile`:
```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Build
RUN npm run build

# Expose port
EXPOSE 3000

# Start
CMD ["npm", "start"]
```

### 3. Run with Docker Compose

Create `docker-compose.yml` in the root:
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend
```

Then run:
```bash
docker-compose up --build
```
