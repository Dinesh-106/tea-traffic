from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import os
from rag_service import RAGService

app = FastAPI(title="Tea Traffic Cafe API")

# CORS Setup
origins = [
    "http://localhost:5173",  # Vite default
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Service
rag_service = RAGService()

# Data Models
class OrderItem(BaseModel):
    id: int
    name: str
    quantity: int
    price: float

class Order(BaseModel):
    customer_name: str
    items: List[OrderItem]
    total_amount: float

class ChatRequest(BaseModel):
    message: str

# Endpoints

@app.get("/")
def read_root():
    return {"message": "Welcome to Tea Traffic Cafe API"}

@app.get("/products")
def get_products():
    try:
        with open("data/menu.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    response = rag_service.query_rag(request.message)
    return {"response": response}

@app.post("/orders")
def create_order(order: Order):
    # In a real app, save to DB. Here we just print or append to a file.
    print(f"New Order from {order.customer_name}: {order}")
    return {"status": "Order received", "order_id": 12345} # Mock ID
