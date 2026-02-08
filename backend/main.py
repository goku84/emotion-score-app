from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import analyzer
import json

app = FastAPI(title="Fake Review Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for the latest analysis
# In a production app, use a database or redis
analysis_result = {
    "summary": None,
    "reviews": None,
    "aspects": None
}

@app.get("/")
def read_root():
    return {"message": "Fake Review Detection API is running"}

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Run the full analysis pipeline
        results = analyzer.analyze_reviews(df)
        
        # Store results in memory
        analysis_result["summary"] = results["summary"]
        analysis_result["reviews"] = results["reviews"]
        analysis_result["aspects"] = results["aspects"]
        
        return {"message": "Analysis complete", "summary": results["summary"]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis-summary")
def get_analysis_summary():
    if not analysis_result["summary"]:
        raise HTTPException(status_code=404, detail="No analysis found. Please upload a dataset first.")
    return analysis_result["summary"]

@app.get("/reviews")
def get_reviews():
    if not analysis_result["reviews"]:
        raise HTTPException(status_code=404, detail="No analysis found")
    return analysis_result["reviews"]

@app.get("/review-details/{review_id}")
def get_review_details(review_id: int):
    if not analysis_result["reviews"]:
        raise HTTPException(status_code=404, detail="No analysis found")
    
    review = next((r for r in analysis_result["reviews"] if r["id"] == review_id), None)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    return review

@app.get("/aspect-report")
def get_aspect_report():
    if not analysis_result["aspects"]:
        raise HTTPException(status_code=404, detail="No analysis found")
    return analysis_result["aspects"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
