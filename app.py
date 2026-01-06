"""
Interactive Chat Interface for RAG Complaint Chatbot
Using Gradio for web interface
"""

import gradio as gr
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Tuple

# Fix import path to find src/
src_dir = os.path.join(os.path.dirname(__file__), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from rag_pipeline import RAGPipeline
    HAS_RAG_PIPELINE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import RAGPipeline: {e}")
    HAS_RAG_PIPELINE = False

# Fallback pipeline (only used if main pipeline fails)
if not HAS_RAG_PIPELINE:
    class RAGPipeline:
        def __init__(self, **kwargs):
            pass
        
        def load_vector_store(self):
            return True
            
        def initialize_llm(self, use_simple=True):
            pass
            
        def run_rag_pipeline(self, question):
            return {
                "question": question,
                "answer": "🔍 This is a demo mode.\n\n"
                         "In full mode, I would analyze real CFPB complaint data to answer your question "
                         "about credit cards, personal loans, savings accounts, or money transfers.",
                "contexts": ["Demo context"],
                "sources": [
                    {"product": "Credit Card", "excerpt_preview": "Example: Unauthorized charges or billing errors..."},
                    {"product": "Personal Loan", "excerpt_preview": "Example: Loan servicing issues or payment problems..."}
                ]
            }

class ComplaintChatbot:
    """Main chatbot application class"""
    
    def __init__(self, vector_store_path: str = "vector_store"):
        self.vector_store_path = vector_store_path
        self.rag_pipeline = None
        self.chat_history = []
        self.initialize_pipeline()
    
    def initialize_pipeline(self):
        """Initialize RAG pipeline with FLAN-T5 for accurate QA."""
        try:
            print("🚀 Initializing RAG Pipeline with FLAN-T5...")
            self.rag_pipeline = RAGPipeline(
                vector_store_path=self.vector_store_path,
                embedding_model_name="all-MiniLM-L6-v2",
                llm_model_name="google/flan-t5-small"  # ✅ Instruction-tuned model
            )
            
            # Load vector store
            vector_store = self.rag_pipeline.load_vector_store()
            if vector_store:
                # Initialize LLM
                self.rag_pipeline.initialize_llm()
                print("✅ RAG Pipeline ready!")
            else:
                raise ValueError("Vector store not loaded")
                
        except Exception as e:
            print(f"⚠️ Error initializing full pipeline: {e}")
            print("🔄 Falling back to demo mode")
            self.rag_pipeline = RAGPipeline()  # Use dummy pipeline

    def process_query(self, query: str, history: List[Tuple] = None) -> Tuple[str, List[Tuple], str]:
        if not query or not query.strip():
            return "", history or [], ""
        
        print(f"💬 Processing: {query}")
        history = history or []
        
        try:
            result = self.rag_pipeline.run_rag_pipeline(query)
            answer = result['answer']
            sources = result.get('sources', [])
            sources_html = self._format_sources_html(sources)
            history.append((query, answer))
            self._log_interaction(query, answer, sources)
            return answer, history, sources_html
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            fallback = ("I apologize, but I couldn't process your question. "
                       "Try asking about credit cards, personal loans, savings accounts, or money transfers.")
            history.append((query, fallback))
            return fallback, history, "<p>⚠️ Error retrieving sources</p>"

    def _format_sources_html(self, sources: List[Dict]) -> str:
        if not sources:
            return "<p>ℹ️ No complaint excerpts found for this query.</p>"
        
        html = "<div style='margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #28a745;'>"
        html += "<h4 style='margin: 0 0 15px 0; color: #155724;'>📚 Sources from Complaint Database</h4>"
        
        for i, source in enumerate(sources[:3]):
            product = source.get('product', 'Unknown')
            excerpt = source.get('excerpt_preview', 'No excerpt')
            html += f"""
            <div style='padding: 10px; margin-bottom: 10px; background: white; border-radius: 4px; border-left: 3px solid #6c757d;'>
                <small style='color: #007bff; font-weight: bold;'>{product}</small>
                <p style='margin: 5px 0; font-size: 0.95em; color: #495057;'>{excerpt}</p>
            </div>
            """
        html += "</div>"
        return html

    def _log_interaction(self, query: str, answer: str, sources: List[Dict]):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer_preview": answer[:100],
            "num_sources": len(sources)
        }
        self.chat_history.append(log_entry)
        if len(self.chat_history) % 5 == 0:
            self._save_chat_log()

    def _save_chat_log(self):
        os.makedirs('data/processed', exist_ok=True)
        try:
            with open('data/processed/chat_log.json', 'w') as f:
                json.dump(self.chat_history, f, indent=2)
        except Exception:
            pass

    def clear_chat(self):
        self.chat_history = []
        return [], ""

def create_gradio_interface():
    chatbot = ComplaintChatbot()
    
    css = """
    .gradio-container { max-width: 900px !important; margin: auto !important; }
    .chatbot { min-height: 450px; max-height: 500px; overflow-y: auto; }
    .sources-container { max-height: 300px; overflow-y: auto; padding: 15px; }
    footer { display: none !important; }
    """
    
    theme = gr.themes.Soft(
        primary_hue="green",
        secondary_hue="blue",
    ).set(
        button_primary_background_fill="#28a745",
        button_primary_background_fill_hover="#218838",
        button_primary_text_color="white",
    )
    
    with gr.Blocks(theme=theme, css=css, title="CrediTrust Complaint Analyst") as demo:
        gr.Markdown("""
        # 🏦 CrediTrust Complaint Analyst
        ### AI-Powered Insights from Real Financial Complaints

        Ask questions about **credit cards**, **personal loans**, **savings accounts**, or **money transfers**.
        Answers are grounded in actual CFPB consumer complaint data.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot_display = gr.Chatbot(
                    label="Chat",
                    elem_classes="chatbot",
                    height=450
                )
                sources_display = gr.HTML(
                    label="Sources",
                    elem_classes="sources-container"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 💡 Examples")
                examples = [
                    "What are common credit card complaints?",
                    "Tell me about personal loan issues",
                    "How do customers complain about savings accounts?",
                    "What problems occur with money transfers?"
                ]
                for q in examples:
                    gr.Button(q, variant="secondary").click(
                        lambda x=q: x, 
                        outputs=query_input
                    )
        
        with gr.Row():
            query_input = gr.Textbox(
                label="Your question",
                placeholder="e.g., What are issues with credit card fees?",
                lines=2,
                scale=4
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        with gr.Row():
            clear_btn = gr.Button("Clear Chat", variant="secondary")
        
        def submit_query(query, history):
            if not query.strip():
                return "", history, ""
            answer, updated_history, sources_html = chatbot.process_query(query, history)
            return "", updated_history, sources_html
        
        query_input.submit(submit_query, [query_input, chatbot_display], [query_input, chatbot_display, sources_display])
        submit_btn.click(submit_query, [query_input, chatbot_display], [query_input, chatbot_display, sources_display])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot_display, sources_display])
        
        gr.Markdown("""
        ---
        <small style="color: #6c757d;">
        <strong>Disclaimer:</strong> Responses are AI-generated based on complaint data. 
        Not financial advice. For official guidance, contact your financial institution.
        </small>
        """)
    
    return demo

def main():
    print("="*80)
    print("🚀 LAUNCHING CREDITRUST COMPLAINT CHATBOT")
    print("="*80)
    print("🔗 Open http://localhost:7860 in your browser")
    
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

if __name__ == "__main__":
    main()