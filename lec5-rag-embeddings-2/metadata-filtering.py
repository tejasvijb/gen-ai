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

def filtered_search(query: str, category: str = None, n_results: int = 3):
    """
    Search with metadata filtering.
    Only retrieve from specific categories.
    """
    query_embedding = get_embedding(query)
    
    # Build where filter
    where_filter = {"category": category} if category else None
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter  # This is the magic!
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

# Demo
print("\n" + "="*60)
print("METADATA FILTERING DEMO")
print("="*60)

query = "How do I change my settings?"

print(f"\n🔍 Query: '{query}'")

print("\n📋 Without filtering (searches everything):")
all_results = semantic_search(query, n_results=3)
for doc, meta in zip(all_results['documents'][0], all_results['metadatas'][0]):
    print(f"  - [{meta['category']}] {doc[:60]}...")

print("\n🔒 With filtering (only 'security' category):")
security_results = filtered_search(query, category="security", n_results=3)
for doc, meta in zip(security_results['documents'][0], security_results['metadatas'][0]):
    print(f"  - [{meta['category']}] {doc[:60]}...")

print("\n💡 Filtering helps when user context narrows the domain!")