# linkedin_assistant.py

import os
import re
import json
import requests
import pandas as pd
import streamlit as st
from typing import List
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# ------------------------------------
# Configuration
# ------------------------------------

PINECONE_API_KEY = "pinecone_api_key_here"  # Replace with your Pinecone API key
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
INDEX_NAME = "langchainvector"
EMBEDDING_DIM = 768

DATA_PATH = r"C:\Users\rajro\some random project\linkedin_companies_dataset.csv"

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "llama2"

# ------------------------------------
# Preprocessing and Chunking
# ------------------------------------

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

def chunk_text(text: str, max_length: int = 300) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i+max_length]) for i in range(0, len(words), max_length)]

def combine_columns(row, columns=['Company', 'Home', 'About', 'Insights']) -> str:
    combined = ''
    for col in columns:
        text = row[col] if pd.notna(row[col]) else ''
        combined += f"{text} "
    return combined.strip()

def preprocess_and_chunk_combined(df: pd.DataFrame, text_column='Combined_Text', chunk_size=300):
    data = []
    for _, row in df.iterrows():
        company = row['Company']
        combined_text = row[text_column]
        if not combined_text.strip():
            continue
        cleaned = clean_text(combined_text)
        chunks = chunk_text(cleaned, max_length=chunk_size)
        for i, chunk in enumerate(chunks):
            data.append({
                'Company': company,
                'Chunk_Index': i,
                'Text_Chunk': chunk
            })
    return pd.DataFrame(data)

# ------------------------------------
# Load Data and Prepare Embeddings
# ------------------------------------

df = pd.read_csv(DATA_PATH)
df['Combined_Text'] = df.apply(combine_columns, axis=1)
chunked_df = preprocess_and_chunk_combined(df)

model = SentenceTransformer("all-mpnet-base-v2")
texts = chunked_df['Text_Chunk'].tolist()
embeddings = model.encode(texts, show_progress_bar=True)
chunked_df['Embedding'] = list(embeddings)

# ------------------------------------
# Pinecone Initialization and Index Setup
# ------------------------------------

pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )
index = pc.Index(INDEX_NAME)

# Upsert vectors to Pinecone
vectors = []
for _, row in chunked_df.iterrows():
    vector_id = f"{row['Company']}_{row['Chunk_Index']}"
    embedding = row['Embedding']
    metadata = {"company": row['Company'], "chunk_index": row['Chunk_Index']}
    vectors.append((vector_id, embedding.tolist(), metadata))
index.upsert(vectors)

# ------------------------------------
# Semantic Search Function
# ------------------------------------

def semantic_search(query: str, top_k: int = 5) -> List[str]:
    query_embedding = model.encode([query])[0]
    result = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    retrieved_texts = []
    for match in result['matches']:
        meta = match['metadata']
        company = meta['company']
        chunk_idx = meta['chunk_index']
        text_chunk = chunked_df[
            (chunked_df['Company'] == company) & (chunked_df['Chunk_Index'] == chunk_idx)
        ]['Text_Chunk'].values[0]
        retrieved_texts.append(text_chunk)
    return retrieved_texts

# ------------------------------------
# Ollama LLaMA 2 Answer Generation
# ------------------------------------

def generate_answer_ollama(question: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"Use the following context to answer the question.\n{context}\n\nQuestion: {question}\nAnswer:"
    response = requests.post(
        OLLAMA_API_URL,
        json={
            "model": OLLAMA_MODEL_NAME,
            "prompt": prompt,
            "options": {"temperature": 0.2, "max_tokens": 256}
        },
        stream=True
    )
    answer = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    answer += data["response"]
            except Exception:
                pass
    return answer.strip()

def answer_query_with_ollama(user_query: str) -> str:
    relevant_chunks = semantic_search(user_query)
    return generate_answer_ollama(user_query, relevant_chunks)

# ------------------------------------
# Streamlit App
# ------------------------------------

st.title("LinkedIn Company Assistant")

user_question = st.text_area("Ask a question about a company:", height=120)

if st.button("Get Answer") and user_question.strip():
    with st.spinner("Searching and generating answer..."):
        try:
            answer = answer_query_with_ollama(user_question)
            st.markdown("### Answer")
            st.write(answer)
            show_context = st.checkbox("Show retrieved context chunks")
            if show_context:
                chunks = semantic_search(user_question)
                for i, chunk in enumerate(chunks):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(chunk)
                    st.markdown("---")
        except Exception as e:
            st.error(f"Error: {e}")
