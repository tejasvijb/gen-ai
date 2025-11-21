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

print("✓ Vector database initialized!")

def get_embedding(text, model="text-embedding-3-small"):
    """Generate embedding for a piece of text."""
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

# Test it
sample_text = "How do I reset my password?"
embedding = get_embedding(sample_text)
print(f"Embedding dimension: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")

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

# Test queries
test_queries = [
    "How can I change my login credentials?",  # Should find password doc
    "I want to remove my profile permanently",  # Should find account deletion
    "How do I get my information from the platform?"  # Should find data export
]

print("\n" + "="*60)
print("SEMANTIC SEARCH DEMO")
print("="*60)

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    print("-" * 60)
    
    results = semantic_search(query, n_results=2)
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        print(f"\nResult {i} (similarity: {1 - distance:.3f}):")
        print(f"Category: {metadata['category']} | Topic: {metadata['topic']}")
        print(f"Content: {doc[:100]}...")
