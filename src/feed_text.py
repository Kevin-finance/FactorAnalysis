from openai import OpenAI
from settings import config

OPENAI_SECRET_KEY = config("OPENAI_SECRET")  
client = OpenAI(api_key = OPENAI_SECRET_KEY)

# We should use Chat.completions for nuanced understanding.

response = client.responses.create(
    model="gpt-4o",
    instructions="You are a coding assistant that talks like a pirate.",
    input = "How do I check if a Python object is an instance of a class?"
)

print(response.output_text)
