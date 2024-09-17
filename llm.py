import os
import openai
from pydantic import BaseModel, Field, ValidationError
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

# Pydantic models for structured responses from the model

class SarcasmDetection(BaseModel):
    """
    Model representing the structure of a sarcasm detection response.
    """
    quote: str = Field(..., description="Detected sarcastic text.")
    score: int = Field(
        ..., 
        ge=0, 
        le=9, 
        description="Score between 0 and 9, where 0 is not sarcastic and 9 is very sarcastic."
    )

class JokeExplanation(BaseModel):
    """
    Model representing the structure of a joke explanation response.
    """
    setup: str = Field(..., description="The initial part of the joke that sets the context.")
    explanation: str = Field(
        ..., 
        description="The explanation of why the joke is funny."
    )
    punchline: str = Field(
        ..., 
        description="The punchline of the joke, delivering the humor."
    )
    joke_type: str = Field(
        None, 
        description="The category or type of the joke (e.g., pun, knock-knock)."
    )
    funny_rating: int = Field(
        None, 
        ge=1, 
        le=10, 
        description="Rating of how funny the joke is on a scale from 1 to 10."
    )

class JokeDelivery(BaseModel):
    """
    Model representing the structure of a joke delivery response.
    """
    setup: str = Field(
        ..., 
        description="The initial part of the joke that sets the context."
    )
    punchline: str = Field(
        ..., 
        description="The punchline of the joke."
    )
    joke_type: str = Field(
        None, 
        description="The category or type of the joke (e.g., pun, knock-knock)."
    )
    funny_rating: int = Field(
        None, 
        ge=1, 
        le=10, 
        description="Rating of how funny the joke is on a scale from 1 to 10."
    )

# System prompt that defines the assistant's persona
SYSTEM_PROMPT_CONTENT = (
    "You are Gorp, The Magnificent. You detect sarcasm, explain jokes, and deliver corny jokes. "
    "Your tone is casual with a touch of whimsy, and you have a fascination with 90s sitcom trivia."
)

def _build_chat_completion_payload(user_message_content: str, existing_messages: list = None):
    """
    Builds the payload required for the chat completion request, including system and user messages,
    and the functions available for the model to call.

    Args:
        user_message_content (str): The content of the user's message.
        existing_messages (list, optional): List of existing messages in the conversation.
    """
    if existing_messages is None:
        existing_messages = []

    # Define the system message and the user's message
    system_message = {"role": "system", "content": SYSTEM_PROMPT_CONTENT}
    user_message = {"role": "user", "content": user_message_content}

    # Combine all messages
    all_messages = [system_message] + existing_messages + [user_message]

    # Define the functions available for the model
    functions = [
        {
            "name": "detect_sarcasm",
            "description": "Detects sarcasm in the given text.",
            "parameters": SarcasmDetection.schema(),
        },
        {
            "name": "explain_joke",
            "description": "Explains a joke provided in the text.",
            "parameters": JokeExplanation.schema(),
        },
        {
            "name": "deliver_joke",
            "description": "Delivers a corny joke.",
            "parameters": JokeDelivery.schema(),
        },
    ]

    return all_messages, functions

DEFAULT_MODEL = "gpt-3.5-turbo"

async def prompt_llm_async(user_message_content: str, existing_messages: list = None, model: str = DEFAULT_MODEL):
    """
    Asynchronously sends a user message to OpenAI's LLM and streams the response.

    Args:
        user_message_content (str): The content of the user's message.
        existing_messages (list, optional): List of existing messages in the conversation.
        model (str, optional): The model to use for the completion.

    Yields:
        str: Chunks of the assistant's response content.
    """
    messages, functions = _build_chat_completion_payload(user_message_content, existing_messages)

    # Call the OpenAI API asynchronously with streaming enabled
    response = await openai.ChatCompletion.acreate(
        model=model,
        messages=messages,
        functions=functions,
        function_call="auto",
        stream=True
    )

    # Stream the response chunks as they become available
    async for chunk in response:
        if 'choices' in chunk and len(chunk['choices']) > 0:
            delta = chunk['choices'][0].get('delta', {})
            if 'content' in delta:
                yield delta['content']

# Function to handle function calls from the model
async def execute_function(function_name: str, arguments: str):
   
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid arguments provided."})

    try:
        if function_name == "detect_sarcasm":
            sarcasm_data = SarcasmDetection(**args)
            # Here you would implement the actual logic for sarcasm detection
            return sarcasm_data.json()
        elif function_name == "explain_joke":
            joke_data = JokeExplanation(**args)
            # Here you would implement the logic to explain the joke
            return joke_data.json()
        elif function_name == "deliver_joke":
            # Generate a joke using the fields defined in JokeDelivery
            joke = JokeDelivery(
                setup="Why did the bicycle fall over?",
                punchline="Because it was two-tired!",
                joke_type="Pun",
                funny_rating=6
            )
            return joke.json()
        else:
            return json.dumps({"error": "Function not found."})
    except ValidationError as ve:
        return json.dumps({"error": ve.errors()})
