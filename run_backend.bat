@echo off
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m uvicorn main:app --reload
