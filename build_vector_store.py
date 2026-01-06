# build_vector_store.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import pandas as pd
import os

# Load sampled data
df = pd.read_csv(r"C:\Users\admin\Rag-complaint-chatbot\data\processed\sampled_complaints.csv")

# Create documents
documents = []
for _, row in df.iterrows():
    text = row.get('cleaned_narrative', '') or row.get('Issue', '')
    if not text or pd.isna(text):
        continue
    metadata = {k: str(v) for k, v in row.items() if k != 'cleaned_narrative'}
    documents.append(Document(page_content=str(text), metadata=metadata))

# Embed and save
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embedding)
vector_store.save_local("vector_store/faiss_index")

print("✅ FAISS index rebuilt locally with compatible versions.")