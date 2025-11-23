from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# --- OPTION 1: ConversationBufferMemory (Current) ---
# Simplest form. Stores the raw text of all messages.
# Pros: Complete context.
# Cons: Can quickly exceed token limits for long conversations.
memory = ConversationBufferMemory(return_messages=True, memory_key="history")

# --- OPTION 2: ConversationBufferWindowMemory ---
# Keeps a list of the last K interactions. Oldest ones are dropped.
# Arguments:
# - k: Number of interactions (pairs of Human/AI messages) to keep.
# from langchain.memory import ConversationBufferWindowMemory
# memory = ConversationBufferWindowMemory(
#     return_messages=True, 
#     memory_key="history", 
#     k=2  # Keeps only the last 2 exchanges
# )

# --- OPTION 3: ConversationTokenBufferMemory ---
# Keeps a buffer of recent interactions, limiting by token count.
# Arguments:
# - max_token_limit: Max number of tokens to store in history.
# - llm: The model used to count tokens (needed to estimate length).
# from langchain.memory import ConversationTokenBufferMemory
# memory = ConversationTokenBufferMemory(
#     return_messages=True, 
#     memory_key="history", 
#     max_token_limit=100, 
#     llm=model
# )

# --- OPTION 4: ConversationSummaryMemory ---
# Uses an LLM to generate a summary of the conversation so far.
# Pros: Can handle very long conversations by condensing context.
# Cons: More expensive (makes an extra LLM call to summarize).
# from langchain.memory import ConversationSummaryMemory
# memory = ConversationSummaryMemory(
#     return_messages=True, 
#     memory_key="history", 
#     llm=model  # Needs LLM to generate summaries
# )

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use conversation history for context."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

# Chain with memory
chain = prompt | model

# TODO: Use langchain to automatically append model response in memory
def chat(user_input):
    # Load memory
    history = memory.load_memory_variables({})["history"]
    
    # Get response
    response = chain.invoke({"input": user_input, "history": history})
    
    # Save to memory
    memory.save_context({"input": user_input}, {"output": response.content})
    
    return response.content

# Test conversation
print(chat("My name is Alex and I'm learning Python."))
print(chat("What's a good project for a beginner?"))
print(chat("What's my name again?"))