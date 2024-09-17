from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from sse_starlette.sse import EventSourceResponse
from typing import Dict, List
from llm import prompt_llm_async  # Removed unused imports

app = FastAPI()

# In-memory dictionary to store chat sessions
chat_sessions: Dict[str, List["Message"]] = {}


class Message(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str


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


@app.post("/chat/{session_id}/message")
async def send_message(session_id: str, message: MessageRequest):
    """
    Sends a message to the model and streams the AI response using SSE.
    """
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Prepare existing messages without the current user's message
    existing_messages = [msg.dict() for msg in chat_sessions[session_id]]

    # Create user's message
    user_message = Message(role="user", content=message.content)
    # Append user's message to history
    chat_sessions[session_id].append(user_message)

    assistant_message = ""

    async def event_generator():
        nonlocal assistant_message
        try:
            # Stream the response from the LLM
            async for chunk in prompt_llm_async(
                user_message_content=message.content,
                existing_messages=existing_messages
            ):
                assistant_message += chunk
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"
        finally:
            # After streaming is complete or an error occurs, append assistant's message to history
            chat_sessions[session_id].append(Message(role="assistant", content=assistant_message))

    return EventSourceResponse(event_generator())


@app.get("/chat/{session_id}/history")
async def get_chat_history(session_id: str):
    """
    Returns the entire message history for a given session.
    """
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"messages": [msg.dict() for msg in chat_sessions[session_id]]}
