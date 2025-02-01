import os
import asyncio
import json
import uvicorn
import requests
from uuid import uuid4
from datetime import datetime
from starlette.middleware.sessions import SessionMiddleware
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from models import QueryRequest
from Services import (
    load_faiss_index,
    load_scheme_data,
    query_faiss_index,
    retrieve_documents,
    generate_response,
    QueryResponse
)

load_dotenv()

# Load environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
WEBHOOK_URL =" https://04cd-2401-4900-57e2-d4c9-6c3b-80d0-dc8c-4b4b.ngrok-free.app" # Your server's public URL

# Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.index = await asyncio.to_thread(load_faiss_index, "faiss_index.bin")
    app.state.scheme_data = await asyncio.to_thread(load_scheme_data, "scheme_details.json")
    app.state.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    app.state.llm_model = "gemini-1.5-pro-002"
    app.state.project_id = os.getenv("GOOGLE_PROJECT_ID")
    
    if not app.state.project_id:
      print("GOOGLE_PROJECT_ID is missing in environment variables!")


    # Set up Telegram Webhook
    set_webhook_response = requests.post(f"{TELEGRAM_API_URL}/setWebhook", json={"url": f"{WEBHOOK_URL}/webhook/query"})
    if not set_webhook_response.json().get("ok"):
        print("Failed to set webhook:", set_webhook_response.json())

    yield  # End of lifespan

# Initialize FastAPI App
app = FastAPI(lifespan=lifespan)

# Telegram Webhook Handler
@app.post("/webhook/query/")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        update = await request.json()

        if "message" in update:
            chat_id = update["message"]["chat"]["id"]
            user_query = update["message"]["text"]

            await process_telegram_query (chat_id, user_query)

        return {"status": "ok"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Process the user query and send response
import logging
logger = logging.getLogger("uvicorn.error")

async def process_telegram_query(chat_id: int, query_text: str):
    try:
        logger.info("Started processing query for chat_id %s: %s", chat_id, query_text)
        
        indices = await asyncio.to_thread(
            query_faiss_index,
            app.state.index,
            query_text,
            app.state.embedding_model
        )
        logger.info("Indices returned: %s", indices)
        
        retrieved_docs = await asyncio.to_thread(
            retrieve_documents,
            indices,
            app.state.scheme_data
        )
        logger.info("Retrieved docs: %s", retrieved_docs)
        logger.info("Calling generate_response with:")
        logger.info("Docs: %s", retrieved_docs)
        logger.info("Model: %s", app.state.llm_model)
        logger.info("Query: %s", query_text)
        logger.info("Project ID: %s", app.state.project_id)

        
        response = await asyncio.to_thread(
        lambda: generate_response(
        retrieved_docs=retrieved_docs,
        model_name=app.state.llm_model,
        query_text=query_text,
        project_id=app.state.project_id
       )
     )

        logger.info("Response generated: %s", response)
        
        message = response.response_text if isinstance(response, QueryResponse) else "Sorry, I couldn't process that."
        logger.info("Final message to send: %s", message)
        
        send_response = requests.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json={"chat_id": chat_id, "text": message}
        )
        logger.info("Telegram sendMessage response: %s", send_response.json())
    
    except Exception as e:
        logger.exception("Error in process_telegram_query:")
        requests.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json={"chat_id": chat_id, "text": "An error occurred!"}
        )


# Fetch chat history (if needed)
@app.get("/getUpdates/")
async def fetch_chat_history():
    response = requests.get(f"{TELEGRAM_API_URL}/getUpdates")
    return response.json()

# Run Application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
