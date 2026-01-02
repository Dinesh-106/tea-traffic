import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("GOOGLE_API_KEY")
SITE_URL = "http://localhost:5173"
APP_NAME = "Tea Traffic Cafe"

class RAGService:
    def __init__(self):
        print("Initializing RAG Service...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
        
        # Initialize OpenRouter Client
        if OPENROUTER_API_KEY:
            self.llm_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
        else:
            self.llm_client = None
            print("Warning: APi Key not found.")

        self.load_knowledge_base()

    def load_knowledge_base(self):
        try:
            with open("data/menu.json", "r") as f:
                menu_items = json.load(f)
            
            self.documents = []

            for item in menu_items:
                doc_text = f"Item: {item['name']}. Description: {item['description']}. Price: {item['price']} rupees. Category: {item['category']}."
                self.documents.append(doc_text)

            # Add extra cafe info
            cafe_info = [
                "Tea Traffic Cafe is open from 8 AM to 10 PM every day.",
                "We are located at 123 Chai Street, Food District.",
                "We specialize in authentic Indian chai and snacks.",
                "Contact us at contact@teatraffic.com."
            ]
            self.documents.extend(cafe_info)

            # Compute embeddings
            print("Computing embeddings...")
            self.embeddings = self.embedding_model.encode(self.documents)
            print("Knowledge base loaded.")
        except Exception as e:
            print(f"Error loading knowledge base: {e}")

    def query_rag(self, user_query: str) -> str:
        if not self.llm_client:
            return "System Error: API Key not configured."

        # 1. Retrieve relevant documents using Cosine Similarity
        query_embedding = self.embedding_model.encode([user_query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top 3 indices
        top_k_indices = np.argsort(similarities)[-3:][::-1]
        
        context_list = [self.documents[i] for i in top_k_indices]
        context = "\n".join(context_list)

        # 2. Generate Response
        system_prompt = """
        You are 'ChaiGPT', a helpful AI assistant for 'Tea Traffic Cafe'.
        Answer based on the context. If unknown, say so politely.
        """
        
        user_message = f"""
        Context:
        {context}

        Customer Query: {user_query}
        """
        
        # List of models to try in order
        models_to_try = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4o-mini",
            "google/gemini-2.0-flash-exp:free",
            "google/gemini-2.0-flash-thinking-exp:free",
            "meta-llama/llama-3-8b-instruct:free",
            "google/gemini-exp-1206:free",
            "google/gemini-pro" # Fallback to standard
        ]

        for model in models_to_try:
            try:
                print(f"Attempting query with model: {model}")
                response = self.llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    extra_headers={
                        "HTTP-Referer": SITE_URL,
                        "X-Title": APP_NAME,
                    }
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Model {model} failed: {e}")
                continue # Try next model
        
        # If all valid models fail
        print("All models failed. Using Mock Response.")
        return "NOTICE: API Connection Failed (Key might be invalid). [MOCK RESPONSE]: We have delicious Samosas (Rs. 20) and Kullad Chai (Rs. 45). Please check your backend .env file to enable real AI."
