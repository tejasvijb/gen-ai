import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

# Define the structure
class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: int = Field(description="Rating out of 10")
    pros: List[str] = Field(description="List of positive points")
    cons: List[str] = Field(description="List of negative points")
    summary: str = Field(description="One line summary")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Construct a prompt that asks for JSON
system_prompt = """You are a movie critic. Analyze the given movie.
Please output the response in valid JSON format with the following keys:
- title: string
- rating: integer (0-10)
- pros: list of strings
- cons: list of strings
- summary: string
"""

movie_name = "Inception"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Review the movie: {movie_name}"}
    ],
    response_format={"type": "json_object"}
)

content = response.choices[0].message.content
data = json.loads(content)

# Parse into object
review = MovieReview(**data)

# Print results
print(f"Title: {review.title}")
print(f"Rating: {review.rating}/10")
print(f"Pros: {review.pros}")
print(f"Cons: {review.cons}")
print(f"Summary: {review.summary}")

