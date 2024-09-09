import os
from openai import OpenAI, Stream
from pydantic import BaseModel, Field

# Initialize OpenAI client with API key from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pydantic models for structured responses from the model
class SarcasmDetection(BaseModel):
    quote: str = Field(..., description="Detected sarcastic text.")
    score: int = Field(..., description="Score between 0 and 9, where 0 is not sarcastic and 9 is very sarcastic.")
    
class JokeExplanation(BaseModel):
    setup: str = Field(..., description="The initial part of the joke that sets the context.")
    premise: str = Field(..., description="The core idea or concept upon which the joke is built.")
    punchline: str = Field(..., description="The punchline of the joke, delivering the humor.")

class JokeDelivery(BaseModel):
    text: str = Field(..., description="The text of the joke.")

# System prompt that defines the assistant's persona
SYSTEM_PROMPT_CONTENT = (
    "You are Gorp, The Magnificent. You detect sarcasm, explain jokes, and deliver corny jokes. "
    "Your tone is casual with a touch of whimsy, and you have a fascination with 90s sitcom trivia."
)

def _build_chat_completion_payload(user_message_content: str, existing_messages: list = None):
    """
    Build the payload required for the chat completion request, including system and user messages.
    """
    if not existing_messages:
        existing_messages = []

    system_message = {"role": "system", "content": SYSTEM_PROMPT_CONTENT}
    user_message = {"role": "user", "content": user_message_content}

    all_messages = [system_message] + existing_messages + [user_message]

    sarcasm_function = {"name": SarcasmDetection.__name__, "parameters": SarcasmDetection.model_json_schema()}
    joke_explanation_function = {"name": JokeExplanation.__name__, "parameters": JokeExplanation.model_json_schema()}
    joke_delivery_function = {"name": JokeDelivery.__name__, "parameters": JokeDelivery.model_json_schema()}

    all_functions = [sarcasm_function, joke_explanation_function, joke_delivery_function]

    return all_messages, all_functions

DEFAULT_MODEL = "gpt-3.5-turbo"

async def prompt_llm_async(user_message_content: str, existing_messages: list = None, model: str = DEFAULT_MODEL):
    """
    Asynchronously send a user message to OpenAI's LLM and get a response.
    """
    messages, functions = _build_chat_completion_payload(user_message_content, existing_messages)
    stream = await client.chat.completions.create(model=model, messages=messages, functions=functions, stream=True)
    return stream
