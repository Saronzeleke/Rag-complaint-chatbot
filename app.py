# app.py
"""
Main application file for the RAG Complaint Chatbot
This can be used to launch either a Gradio or Streamlit interface
"""

import sys
import os
from pathlib import Path

def run_task1():
    """Run Task 1: Data Preprocessing"""
    print("Running Task 1: Data Preprocessing...")
    from src.data_preprocessing import main as task1_main
    task1_main()

def run_task2():
    """Run Task 2: Embedding Pipeline"""
    print("Running Task 2: Embedding Pipeline...")
    from src.embedding_pipeline import main as task2_main
    task2_main()

def setup_environment():
    """Setup project environment"""
    print("Setting up project environment...")
    
    # Create necessary directories
    directories = [
        'data/raw',
        'data/processed',
        'vector_store',
        'notebooks',
        'src',
        'tests'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("Environment setup complete!")

def main():
    """Main application entry point"""
    print("=" * 60)
    print("RAG COMPLAINT CHATBOT PROJECT")
    print("=" * 60)
    
    while True:
        print("\nSelect an option:")
        print("1. Setup project environment")
        print("2. Run Task 1: Data Preprocessing")
        print("3. Run Task 2: Embedding Pipeline")
        print("4. Run both tasks sequentially")
        print("5. Launch Notebooks")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            setup_environment()
        
        elif choice == '2':
            run_task1()
        
        elif choice == '3':
            run_task2()
        
        elif choice == '4':
            setup_environment()
            run_task1()
            run_task2()
        
        elif choice == '5':
            print("\nAvailable notebooks:")
            print("1. notebooks/eda_preprocessing.ipynb - Task 1: EDA and Preprocessing")
            print("2. notebooks/embedding_pipeline.ipynb - Task 2: Embedding Pipeline")
            print("\nOpen these in Jupyter Notebook or JupyterLab")
        
        elif choice == '6':
            print("\nExiting... Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()