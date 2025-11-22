import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.Client()

# Create a collection (like a table in SQL)
collection = chroma_client.create_collection(
    name="documentation",
    metadata={"description": "Product documentation embeddings"}
)

def get_embedding(text, model="text-embedding-3-small"):
    """Generate embedding for a piece of text."""
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

def semantic_search(query, n_results=3):
    """Search for documents similar to the query."""
    # Generate embedding for the query
    query_embedding = get_embedding(query)
    
    # Search in vector database
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return results

# Sample documentation
documents = [
    "To reset your password, go to Settings > Security > Change Password. Enter your current password and then your new password twice.",
    "You can update your email address in the Account section. Click on Profile, then Edit Email, and verify the change via the confirmation link.",
    "To delete your account, navigate to Settings > Privacy > Delete Account. This action is permanent and cannot be undone.",
    "Enable two-factor authentication in Security settings. You'll need a mobile app like Google Authenticator or Authy.",
    "Export your data by going to Settings > Data & Privacy > Download Data. Processing may take up to 48 hours.",
    "Change your username in Profile settings. Note that usernames must be unique and can only be changed once every 30 days.",
    "To recover a deleted item, check your Trash folder within 30 days. After 30 days, items are permanently removed.",
    "Manage notification preferences in Settings > Notifications. You can customize alerts for email, push, and SMS."
]

# Metadata for each document
metadata = [
    {"category": "security", "topic": "password"},
    {"category": "account", "topic": "email"},
    {"category": "account", "topic": "deletion"},
    {"category": "security", "topic": "2fa"},
    {"category": "privacy", "topic": "data-export"},
    {"category": "account", "topic": "username"},
    {"category": "recovery", "topic": "trash"},
    {"category": "settings", "topic": "notifications"}
]

print("Generating embeddings and storing documents...")

# Generate embeddings for all documents
embeddings = [get_embedding(doc) for doc in documents]

# Store in vector database
collection.add(
    embeddings=embeddings,
    documents=documents,
    metadatas=metadata,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print(f"✓ Stored {len(documents)} documents in vector database!")

def rag_query(user_question, n_results=3):
    """
    Complete RAG pipeline:
    1. Take user question
    2. Generate embedding
    3. Retrieve relevant documents
    4. Create context
    5. Generate answer with LLM
    """
    
    # Step 1 & 2: Embed and retrieve
    print(f"🔍 Searching knowledge base for: '{user_question}'")
    results = semantic_search(user_question, n_results=n_results)
    
    # Step 3: Build context from retrieved documents
    context_docs = results['documents'][0]
    context = "\n\n".join([f"Document {i+1}: {doc}" 
                          for i, doc in enumerate(context_docs)])
    
    print(f"✓ Found {len(context_docs)} relevant documents")
    
    # Step 4: Create prompt with context
    system_prompt = """You are a helpful assistant. Answer the user's question 
using ONLY the information provided in the context below. If the answer cannot 
be found in the context, say so clearly. Do not make up information."""
    
    user_prompt = f"""Context:
{context}

Question: {user_question}

Answer:"""
    
    # Step 5: Generate answer
    print("🤖 Generating answer...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    answer = response.choices[0].message.content
    
    return {
        "answer": answer,
        "sources": context_docs,
        "metadata": results['metadatas'][0]
    }

# Test the complete RAG system
print("\n" + "="*60)
print("COMPLETE RAG SYSTEM DEMO")
print("="*60)

test_question = "What should I do if I accidentally deleted something important?"

result = rag_query(test_question)

print("\n" + "-"*60)
print(f"Question: {test_question}")
print("-"*60)
print(f"\nAnswer:\n{result['answer']}")
print("\n📚 Sources used:")
for i, (source, meta) in enumerate(zip(result['sources'], result['metadata']), 1):
    print(f"\n{i}. [{meta['category']}] {source[:80]}...")