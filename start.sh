#!/bin/bash
# Start FastAPI (Uvicorn) in the background
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
# Start Streamlit app in the foreground
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 