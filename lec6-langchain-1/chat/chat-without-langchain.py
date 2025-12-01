import os
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI

load_dotenv()

print("\n--- Anthropic Claude ---")
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

messages = [
    {"role": "user", "content": "Explain semantic search in two sentences."}
]

resp = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=120,
    temperature=0.2,
    system="You are a helpful assistant that replies concisely.",
    messages=messages
)

print(resp.content[0].text)



print("\n--- OpenAI ---")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

messages = [
    {"role": "system", "content": "You are a helpful assistant that replies concisely."},
    {"role": "user", "content": "Explain semantic search in two sentences."}
]

resp = client.chat.completions.create(
    model="gpt-4o-mini",   # pick a model available to you
    messages=messages,
    temperature=0.2,
    max_tokens=120
)

print(resp.choices[0].message.content)
