#!/usr/bin/env python3
"""
Full RAG Pipeline for CFPB Complaints (Tasks 1-4)
Author: Data Scientist
"""

import pandas as pd
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import pickle
import gradio as gr

# ----------------------------
# CONFIGURATION
# ----------------------------
INPUT_PATH = r"C:\Users\admin\Rag-complaint-chatbot\data\raw\full_complaints.csv"
FILTERED_PATH = r"C:\Users\admin\Rag-complaint-chatbot\data\filtered_complaints.csv"
VECTOR_STORE_DIR = Path("vector_store")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SAMPLE_SIZE = 50000  # can adjust to 10k-50k
TOP_K = 5  # top-k retrieval

PRODUCTS_TO_KEEP = [
    'Credit card',
    'Personal loan',
    'Savings account',
    'Money transfers',
    'Checking account'  # Example 5th product
]

BOILERPLATE_PATTERNS = [
    r"i am writing to file a complaint",
    r"i would like to complain",
    r"please investigate",
    r"this complaint is regarding"
]

# ----------------------------
# UTILITY FUNCTIONS
# ----------------------------
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower()
    for pat in BOILERPLATE_PATTERNS:
        text = re.sub(pat, "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_filter_dataset(input_path: str) -> pd.DataFrame:
    """Load large CSV in chunks, filter products, remove empty narratives, clean text."""
    chunks = []
    for chunk in pd.read_csv(input_path, chunksize=500_000):
        chunk = chunk[chunk['Product'].isin(PRODUCTS_TO_KEEP)]
        chunk = chunk[chunk['Consumer complaint narrative'].notnull()]
        chunk['Consumer complaint narrative'] = chunk['Consumer complaint narrative'].apply(clean_text)
        # Keep only necessary columns
        cols_to_keep = ['Product', 'Consumer complaint narrative']
        if 'Complaint ID' in chunk.columns:
            cols_to_keep.append('Complaint ID')
        chunk = chunk[cols_to_keep]
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    return df

def stratified_sample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Stratified sampling by product."""
    frac = min(1, sample_size / len(df))
    df_sample = df.groupby('Product', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=42))
    return df_sample

# ----------------------------
# TASK 1: EDA & Preprocessing
# ----------------------------
print("[Task 1] Loading and cleaning dataset...")
df = load_and_filter_dataset(INPUT_PATH)
print(f"Total complaints after filtering: {len(df)}")
df_sample = stratified_sample(df, SAMPLE_SIZE)
Path(FILTERED_PATH).parent.mkdir(parents=True, exist_ok=True)
df_sample.to_csv(FILTERED_PATH, index=False)
print(f"Filtered dataset saved to: {FILTERED_PATH}")

# ----------------------------
# TASK 2: Chunking & Embeddings
# ----------------------------
print("[Task 2] Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

all_chunks = []
all_metadata = []

for idx, row in df_sample.iterrows():
    chunks = text_splitter.split_text(row['Consumer complaint narrative'])
    all_chunks.extend(chunks)
    all_metadata.extend([{
        'complaint_id': row.get('Complaint ID', idx),
        'product': row['Product']
    }] * len(chunks))

print(f"Total chunks created: {len(all_chunks)}")

print("[Task 2] Generating embeddings...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = embedding_model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
VECTOR_STORE_DIR.mkdir(exist_ok=True)
faiss.write_index(index, VECTOR_STORE_DIR / "faiss_index.idx")
with open(VECTOR_STORE_DIR / "metadata.pkl", "wb") as f:
    pickle.dump(all_metadata, f)
print(f"Vector store saved at: {VECTOR_STORE_DIR}")

# ----------------------------
# TASK 3: RAG Core Logic
# ----------------------------
# Load index & metadata (for later use in interface)
index = faiss.read_index(str(VECTOR_STORE_DIR / "faiss_index.idx"))
with open(VECTOR_STORE_DIR / "metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def retrieve(query: str, top_k: int = TOP_K):
    """Retrieve top-k complaint chunks for a query."""
    q_vec = embedding_model.encode([query])
    D, I = index.search(q_vec, top_k)
    retrieved = [metadata[i] for i in I[0]]
    return retrieved

# ----------------------------
# TASK 4: Gradio Interface
# ----------------------------
def rag_chat(query: str):
    results = retrieve(query)
    context = "\n".join([f"Product: {r['product']}, Complaint ID: {r['complaint_id']}" for r in results])
    # Placeholder: Replace with actual LLM integration if desired
    answer = f"(LLM would generate answer here based on context)"
    return f"Retrieved context:\n{context}\n\nAnswer:\n{answer}"

iface = gr.Interface(
    fn=rag_chat,
    inputs="text",
    outputs="text",
    title="CFPB Complaint RAG Chatbot",
    description="Ask questions about customer complaints. Returns retrieved complaints and a placeholder answer."
)

print("[Task 4] Launching Gradio interface...")
iface.launch()
