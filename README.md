# Fake Review Detection & Explainable Opinion Mining

This project is a web application for detecting fake reviews and analyzing sentiment/emotions in product reviews.

## Installation

### Prerequisites
- Python 3.8+
- Node.js 18+

### Backend Setup

1.  Navigate to the project root.
2.  Install Python dependencies:
    ```bash
    pip install -r backend/requirements.txt
    ```
3.  Download Spacy model:
    ```bash
    python -m spacy download en_core_web_sm
    ```
4.  Run the backend server:
    ```bash
    uvicorn backend.main:app --reload --port 8000
    ```

### Frontend Setup

1.  Navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Run the development server:
    ```bash
    npm run dev
    ```

## Usage

1.  Open the frontend at `http://localhost:3000`.
2.  Upload a CSV file containing review data. The CSV should have a `Text` column (and optionally `Rating`, `Summary`, `Time`, `UserId`, `ProductId`).
3.  View the analysis results, including trust scores, fake review count, emotion analysis, and aspect-based summaries.

## Features

-   **Fake Review Detection**: Uses an Autoencoder model and text/metadata features to detect anomalies.
-   **Opinion Mining**: Extracts aspects (Battery, Camera, etc.) and analyzes sentiment per aspect.
-   **Emotion Analysis**: Detects emotions (Joy, Anger, etc.) using RoBERTa.
-   **Explainability**: Provides reasons for flagging reviews as fake.

## Project Structure

-   `backend/`: FastAPI backend and ML logic.
    -   `main.py`: API endpoints.
    -   `analyzer.py`: Core logic for review analysis.
-   `frontend/`: Next.js frontend.
    -   `app/`: Pages and layouts.
    -   `components/`: React components.
