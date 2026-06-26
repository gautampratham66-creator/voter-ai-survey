"""
rag/rag_system.py
Retrieval Augmented Generation using LangChain + ChromaDB + Claude
Allows natural language querying of voter survey database
"""

# pip install chromadb langchain langchain-anthropic sentence-transformers

import chromadb
from chromadb.utils import embedding_functions
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import json

# ─── ChromaDB Setup ──────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="voter_survey",
    embedding_function=embedding_fn
)

# ─── Index Survey Data into ChromaDB ────────────────────────────────────────
def index_family_data(families: list):
    """
    Converts family records into text documents and stores in ChromaDB.
    Called whenever new survey data is added.
    """
    documents = []
    metadatas = []
    ids = []

    for fam in families:
        for member in fam["members"]:
            doc_text = f"""
            Family ID: {fam['family_id']}
            Head: {fam['head']}
            Village: {fam.get('village', 'N/A')}
            District: {fam['district']}
            Member Name: {member['name']}
            Age: {member['age']}
            Gender: {member['gender']}
            Eligible for Voter ID: {'Yes' if member['eligible'] else 'No (under 18)'}
            Has Voter ID: {'Yes' if member['has_voter_id'] else 'No'}
            Status: {'Registered' if member['has_voter_id'] else ('Missing - Needs Registration' if member['eligible'] else 'Not Eligible')}
            """
            uid = f"{fam['family_id']}_{member['name'].replace(' ','_')}"
            documents.append(doc_text.strip())
            metadatas.append({
                "family_id": fam['family_id'],
                "district": fam['district'],
                "gender": member['gender'],
                "has_voter_id": str(member['has_voter_id']),
                "eligible": str(member['eligible'])
            })
            ids.append(uid)

    # Upsert into ChromaDB (add or update)
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print(f"[RAG] Indexed {len(documents)} member records into ChromaDB")

# ─── RAG Query Function ──────────────────────────────────────────────────────
def rag_query(question: str, top_k: int = 5) -> str:
    """
    Main RAG function:
    1. Embeds the question
    2. Retrieves top-k relevant records from ChromaDB
    3. Sends context + question to Claude
    4. Returns grounded answer
    """

    # Step 1: Retrieve from ChromaDB
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    retrieved_docs = results['documents'][0] if results['documents'] else []

    # Step 2: Build context
    context = "\n\n---\n".join(retrieved_docs) if retrieved_docs else "No relevant records found."

    # Step 3: Prompt Claude with context
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        anthropic_api_key="YOUR_ANTHROPIC_API_KEY"
    )

    prompt = f"""You are an AI analyst for a government Voter ID Survey system.
    
Answer the following question using ONLY the provided survey data context.
Be specific, use numbers, and highlight any gender or district disparities.

SURVEY DATA CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    response = llm.invoke(prompt)
    return response.content

# ─── Filtered Queries (by district/gender) ──────────────────────────────────
def query_by_district(district: str) -> list:
    """Retrieve all records for a specific district"""
    results = collection.get(
        where={"district": district}
    )
    return results['documents']

def query_missing_voters(gender: str = None) -> list:
    """Retrieve all members who are eligible but missing Voter ID"""
    where_clause = {"has_voter_id": "False", "eligible": "True"}
    if gender:
        where_clause["gender"] = gender
    results = collection.get(where=where_clause)
    return results['documents']

# ─── Example Usage ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Sample data
    sample_families = [
        {"family_id":"F001","head":"Ramesh Kumar","village":"Kairana","district":"Shamli","members":[
            {"name":"Ramesh Kumar","age":45,"gender":"Male","has_voter_id":True,"eligible":True},
            {"name":"Sunita Kumar","age":41,"gender":"Female","has_voter_id":False,"eligible":True},
        ]}
    ]

    # Index data
    index_family_data(sample_families)

    # Query
    answer = rag_query("How many females are missing their Voter ID?")
    print("\n=== RAG ANSWER ===")
    print(answer)
