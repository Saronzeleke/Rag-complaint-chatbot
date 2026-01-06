# RAG Complaint Chatbot 🏦

A Retrieval-Augmented Generation (RAG) chatbot for analyzing consumer financial complaints from the CFPB database.

## 🎯 Project Overview

This project implements a complete RAG system that can:

- Analyze consumer complaints about financial products

- Retrieve relevant complaint excerpts based on user queries

- Generate informed responses using LLMs

- Provide source attribution for transparency

## 📋 Project Structure

rag-complaint-chatbot/

├── data/

│ ├── raw/ # Original complaint data

│ └── processed/ # Processed datasets and reports

├── vector_store/ # FAISS vector embeddings

├── notebooks/ # Jupyter notebooks for EDA

├── src/ # Source code modules

│ ├── data_preprocessing.py # Task 1: EDA & preprocessing

│ ├── embedding_pipeline.py # Task 2: Chunking & embedding

│ ├── rag_pipeline.py # Task 3: RAG core logic

│ └── init.py

├── tests/ # Unit tests
    test_all_tasks.py # Comprehensive validation

├── app.py # Gradio/Streamlit interface

├── requirements.txt # Dependencies


└── README.md

text

## 🚀 Quick Start

### 1. Installation

# Clone repository

git clone https://github.com/Saronzeleke/Rag-complaint-chatbot.git

cd Rag-complaint-chatbot

# Install dependencies

pip install -r requirements.txt

### 2. Data Setup

Place your complaints.csv file in data/raw/ or update the path in src/data_preprocessing.py.

## 3. Run All Tasks

bash

# Run Task 1: Data preprocessing

python src/data_preprocessing.py

# Run Task 2: Vector store creation

python src/embedding_pipeline.py

# Run Task 3: RAG evaluation

python src/rag_pipeline.py

# Run Task 4: Launch chatbot

python app.py

Or run validation to check all tasks:

python test_all_tasks.py

📊 Task Implementation

Task 1: EDA & Data Preprocessing

✅ Loads CFPB complaint dataset

✅ Analyzes product distributions and narrative lengths

✅ Filters for 4 target financial products

✅ Cleans text narratives

✅ Generates EDA reports and visualizations

Task 2: Vector Store Setup

✅ Stratified sampling (12,000 complaints)

✅ Text chunking with optimal parameters

✅ Embeddings using all-MiniLM-L6-v2

✅ FAISS vector store with metadata

✅ Persisted for reuse

Task 3: RAG Core Logic

✅ Semantic retrieval with similarity search

✅ Prompt engineering for financial analysis

✅ LLM integration (DialoGPT/GPT-2)

✅ Qualitative evaluation with 10 test questions

✅ Comprehensive evaluation reports

Task 4: Interactive Interface

✅ Gradio web interface

✅ Real-time query processing

✅ Source attribution display

✅ Chat history management

✅ Clean, intuitive UI

🔧 Technical Details

Models Used

Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional)

LLM: microsoft/DialoGPT-small (or GPT-2 fallback)

Vector Store: FAISS for efficient similarity search

Key Features

Stratified Sampling: Balanced representation across products

Intelligent Chunking: 500 chars with 50 overlap for optimal context

Source Attribution: Shows which complaints informed each answer

Error Handling: Robust with fallback mechanisms

Modular Design: Clean separation of concerns

📈 Evaluation Results

The system was evaluated on 10 representative questions:

Question	Quality Score	Retrieval Score

Credit card issues	4/5	0.85

Personal loan complaints	4/5	0.80

Savings account problems	3/5	0.75

Money transfer concerns	4/5	0.90

Average Quality Score: 3.8/5

Average Retrieval Score: 0.82/1.0

🎨 Interface Features

Clean Chat Interface: User-friendly conversation flow

Source Display: Shows retrieved complaint excerpts

Example Questions: Quick-start prompts

Clear Functionality: Reset conversation easily

Responsive Design: Works on different screen sizes

📝 Requirements Met

Core Requirements

✅ Loads and preprocesses complaint data

✅ Creates stratified sample with chunking

✅ Builds and persists vector store

✅ Implements RAG retrieval and generation

✅ Provides qualitative evaluation

✅ Creates interactive chat interface

✅ Displays sources for transparency

Best Practices

✅ Modular, documented code

✅ Proper project structure

✅ Git & GitHub workflows

✅ Error handling and logging

✅ Comprehensive testing

🚦 Running the Application

Gradio Interface (Default)

python app.py

# Open http://localhost:7860 in browser

Streamlit Interface

streamlit run app.py -- --ui streamlit

With Public Sharing (Gradio)

python app.py --share

📁 Output Files

data/processed/filtered_complaints.csv - Cleaned dataset

data/processed/eda_report.txt - EDA analysis

vector_store/ - FAISS index and metadata

data/processed/rag_evaluation.csv - RAG evaluation results

data/processed/validation_report.json - Task validation

🔍 Testing

Run comprehensive validation:

python test_all_tasks.py

Run unit tests:

pytest tests/

🛠️ Customization

Change Target Products

Edit target_products in src/data_preprocessing.py:

python
self.target_products = [
    'Credit card', 
    'Personal loan', 
    'Savings account', 
    'Money transfers'
]

Adjust Chunking Parameters

Edit in src/embedding_pipeline.py:

python

chunks = pipeline.chunk_texts(chunk_size=500, chunk_overlap=50)
Use Different LLM
Update in src/rag_pipeline.py:

python

rag_pipeline = RAGPipeline(llm_model_name="your-model-here")

🤝 Contributing

Fork the repository

Create a feature branch

Commit changes

Push to branch

Create Pull Request

📄 License

This project is for educational purposes as part of a data science assignment.

🙏 Acknowledgments

CFPB for the complaint dataset

Hugging Face for transformer models

LangChain for RAG framework

FAISS for vector similarity search