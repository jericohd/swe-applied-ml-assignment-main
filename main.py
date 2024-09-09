import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from llm import prompt_llm_async, SarcasmDetection, JokeExplanation, JokeDelivery

app = FastAPI()

# In-memory dictionary to store chat sessions
chat_sessions = {}

# Pydantic model to handle incoming user messages
class MessageRequest(BaseModel):
    content: str

@app.post("/chat/session")
async def create_chat_session():
    """
    Creates a new chat session and returns a unique session ID.
    """
    session_id = str(uuid4())
    chat_sessions[session_id] = []  # Initialize with an empty history
    return {"session_id": session_id}

@app.post("/chat/{session_id}/sarcasm")
async def detect_sarcasm(session_id: str, message: MessageRequest):
    """
    Sends a message to the model and detects sarcasm in the content.
    """
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    chat_sessions[session_id].append({"role": "user", "content": message.content})
    
    response = await prompt_llm_async(user_message_content=message.content, existing_messages=chat_sessions[session_id])
    
    sarcasm_detection = SarcasmDetection(**response)
    
    chat_sessions[session_id].append({"role": "assistant", "content": sarcasm_detection.quote})

    return {
        "sarcasm_detected": sarcasm_detection.quote,
        "score": sarcasm_detection.score
    }

@app.post("/chat/{session_id}/joke_explanation")
async def explain_joke(session_id: str, message: MessageRequest):
    """
    Sends a joke to the model and gets an explanation for it.
    """
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    chat_sessions[session_id].append({"role": "user", "content": message.content})
    
    response = await prompt_llm_async(user_message_content=message.content, existing_messages=chat_sessions[session_id])
    
    joke_explanation = JokeExplanation(**response)

    chat_sessions[session_id].append({"role": "assistant", "content": f"{joke_explanation.setup} - {joke_explanation.premise} - {joke_explanation.punchline}"})

    return {
        "setup": joke_explanation.setup,
        "premise": joke_explanation.premise,
        "punchline": joke_explanation.punchline
    }

@app.get("/chat/{session_id}/joke_delivery")
async def deliver_joke(session_id: str):
    """
    Requests the model to deliver a joke.
    """
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    response = await prompt_llm_async(user_message_content="Tell me a joke", existing_messages=chat_sessions[session_id])
    
    joke_delivery = JokeDelivery(**response)

    chat_sessions[session_id].append({"role": "assistant", "content": joke_delivery.text})

    return {"joke": joke_delivery.text}
