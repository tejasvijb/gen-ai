import random
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv

load_dotenv()

# ========== STEP 1: Define Tools with @tool decorator ==========
# LangChain automatically generates the JSON schema from docstrings and type hints!

@tool
def get_weather(location: str, unit: str = "fahrenheit") -> dict:
    """Get current weather for a location.
    
    Args:
        location: City and state, e.g., 'Seattle, WA'
        unit: Temperature unit - 'celsius' or 'fahrenheit'
    """
    temp = random.randint(60, 85) if unit == "fahrenheit" else random.randint(15, 30)
    return {
        "location": location,
        "temperature": temp,
        "unit": unit,
        "conditions": random.choice(["sunny", "cloudy", "rainy"])
    }

@tool
def calculate(expression: str) -> dict:
    """Perform mathematical calculations.
    
    Args:
        expression: Mathematical expression to evaluate, e.g., '2+2' or '15*23'
    """
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid expression"}
        result = eval(expression)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_flight_info(origin: str, destination: str, date: str = None) -> dict:
    """Get flight information between two cities.
    
    Args:
        origin: Origin city
        destination: Destination city  
        date: Travel date in YYYY-MM-DD format
    """
    return {
        "origin": origin,
        "destination": destination,
        "date": date or "2025-01-15",
        "price": random.randint(200, 800),
        "duration": f"{random.randint(2, 8)}h {random.randint(0, 59)}m",
        "airline": random.choice(["Delta", "United", "American"])
    }

# ========== STEP 2: Create Agent with Tools ==========

tools = [get_weather, calculate, get_flight_info]
model = ChatOpenAI(model="gpt-4o", temperature=0)

# create_react_agent handles the entire tool-calling loop automatically!
agent = create_react_agent(model, tools)

# ========== STEP 3: Run Complex Query ==========

def run_conversation(user_message: str):
    """Run a conversation that may need multiple tool calls."""
    print(f"User: {user_message}\n")
    print("=" * 50)
    
    # The agent handles everything: tool selection, execution, looping
    result = agent.invoke({"messages": [HumanMessage(content=user_message)]})
    
    # Print the conversation flow
    for msg in result["messages"]:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"🔧 Tool Call: {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            print(f"   ✓ Result: {msg.content[:100]}...")
        elif hasattr(msg, 'content') and msg.content and not isinstance(msg, HumanMessage):
            print(f"\nAssistant: {msg.content}")
    
    return result["messages"][-1].content

# Demo
print("=" * 70)
print("LANGCHAIN DEMO: Multiple Tools with Native Tool Calling")
print("=" * 70)

run_conversation(
    "I'm planning a trip from Seattle to Miami. What's the weather like in Miami, "
    "and can you find me a flight? Also calculate the total cost if I need to book 3 tickets."
)