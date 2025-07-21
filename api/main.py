import os
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Load model, matrix, and summaries
with open("models/knn_model.pkl", "rb") as f:
    model, user_product_matrix, top_summaries = pickle.load(f)

API_KEY = os.getenv("API_KEY", "changeme123")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

class UserRequest(BaseModel):
    user_id: str

@app.post("/recommend")
def recommend(req: UserRequest, api_key: str = Depends(verify_api_key)):
    user_id = req.user_id

    if user_id not in user_product_matrix.index:
        raise HTTPException(status_code=404, detail="User not found")

    user_vector = user_product_matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = model.kneighbors(user_vector, n_neighbors=6)

    recommended_items = set()
    for i in indices[0]:
        sim_user = user_product_matrix.index[i]
        if sim_user != user_id:
            top_items = user_product_matrix.loc[sim_user].sort_values(ascending=False).head(5).index
            recommended_items.update(top_items)

    output = []
    for pid in list(recommended_items)[:5]:
        summary = top_summaries.get(pid, "No summary available")
        output.append({"product_id": pid, "summary": summary})

    return {"user_id": user_id, "recommendations": output}

@app.get("/users")
def get_users(api_key: str = Depends(verify_api_key)):
    """Get list of available user IDs"""
    return {"users": list(user_product_matrix.index[:10])}  # Show first 10 users

@app.get("/")
def root(api_key: str = Depends(verify_api_key)):
    """Root endpoint"""
    return {"message": "ML Recommendation API", "endpoints": ["/recommend", "/users", "/docs"]}