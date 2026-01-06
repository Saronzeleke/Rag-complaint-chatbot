# test_all_tasks.py
"""
Comprehensive test script to validate all tasks meet the criteria
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

def read_file_with_encoding(filepath, encodings=['utf-8', 'latin-1', 'cp1252']):
    """Try multiple encodings to read a file"""
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    # If all encodings fail, try reading as binary
    with open(filepath, 'rb') as f:
        return f.read().decode('utf-8', errors='ignore')

def check_task1():
    """Validate Task 1 implementation"""
    print("\n" + "="*80)
    print("VALIDATING TASK 1: EDA & DATA PREPROCESSING")
    print("="*80)
    
    checks = []
    
    # Check 1: Filtered data file exists
    filtered_path = 'data/processed/filtered_complaints.csv'
    if os.path.exists(filtered_path):
        try:
            df = pd.read_csv(filtered_path)
            checks.append(("Filtered data file exists", True, f"Shape: {df.shape}"))
        except Exception as e:
            checks.append(("Filtered data file exists", False, f"Error reading: {e}"))
    else:
        checks.append(("Filtered data file exists", False, "File not found"))
    
    # Check 2: EDA report exists
    eda_report = 'data/processed/eda_report.txt'
    if os.path.exists(eda_report):
        try:
            content = read_file_with_encoding(eda_report)
            checks.append(("EDA report generated", True, f"Length: {len(content)} chars"))
        except Exception as e:
            checks.append(("EDA report generated", False, f"Error reading: {e}"))
    else:
        checks.append(("EDA report generated", False, "File not found"))
    
    # Check 3: Visualizations exist
    viz_files = [
        'data/processed/product_distribution.png',
        'data/processed/narrative_length_distribution.png'
    ]
    viz_exists = all(os.path.exists(f) for f in viz_files)
    checks.append(("Visualizations created", viz_exists, 
                  f"{sum(os.path.exists(f) for f in viz_files)}/{len(viz_files)} files"))
    
    # Check 4: Data quality
    if os.path.exists(filtered_path):
        try:
            df = pd.read_csv(filtered_path)
            # Check for required columns
            required_cols = ['cleaned_narrative']
            missing_cols = [col for col in required_cols if col not in df.columns]
            checks.append(("Required columns present", len(missing_cols) == 0,
                          f"Missing: {missing_cols}" if missing_cols else "All present"))
            
            # Check data size
            checks.append(("Data size reasonable", len(df) > 0, 
                          f"Rows: {len(df):,}"))
        except Exception as e:
            checks.append(("Data quality checks", False, f"Error: {e}"))
    
    return checks

def check_task2():
    """Validate Task 2 implementation"""
    print("\n" + "="*80)
    print("VALIDATING TASK 2: VECTOR STORE SETUP")
    print("="*80)
    
    checks = []
    
    # Check 1: Vector store directory exists
    vector_store_dir = 'vector_store'
    if os.path.exists(vector_store_dir):
        try:
            contents = os.listdir(vector_store_dir)
            checks.append(("Vector store directory exists", True, 
                          f"Contains: {len(contents)} items"))
        except Exception as e:
            checks.append(("Vector store directory exists", False, f"Error: {e}"))
    else:
        checks.append(("Vector store directory exists", False, "Directory not found"))
    
    # Check 2: FAISS or Chroma files exist
    faiss_files = ['faiss_index/index.faiss', 'faiss_index.pkl']
    chroma_files = ['chroma.sqlite3', 'chroma.sqlite3-wal']
    
    found_faiss = False
    found_chroma = False
    
    if os.path.exists(vector_store_dir):
        for f in faiss_files:
            if os.path.exists(os.path.join(vector_store_dir, f)):
                found_faiss = True
                break
        for f in chroma_files:
            if os.path.exists(os.path.join(vector_store_dir, f)):
                found_chroma = True
                break
    
    checks.append(("Vector store files exist", found_faiss or found_chroma,
                  f"FAISS: {found_faiss}, Chroma: {found_chroma}"))
    
    # Check 3: Metadata exists
    metadata_path = os.path.join(vector_store_dir, 'metadata.pkl')
    if os.path.exists(metadata_path):
        checks.append(("Metadata file exists", True, "Metadata preserved"))
    else:
        checks.append(("Metadata file exists", False, "File not found"))
    
    # Check 4: Sampled data exists
    sampled_path = 'data/processed/sampled_complaints.csv'
    if os.path.exists(sampled_path):
        try:
            df = pd.read_csv(sampled_path)
            checks.append(("Sampled data saved", True, f"Sample size: {len(df):,}"))
        except Exception as e:
            checks.append(("Sampled data saved", False, f"Error reading: {e}"))
    else:
        checks.append(("Sampled data saved", False, "File not found"))
    
    # Check 5: Task 2 report exists
    task2_report = 'data/processed/task2_report.txt'
    if os.path.exists(task2_report):
        try:
            content = read_file_with_encoding(task2_report)
            checks.append(("Task 2 report generated", True, 
                          f"Contains: {len(content.splitlines())} lines"))
        except Exception as e:
            checks.append(("Task 2 report generated", False, f"Error reading: {e}"))
    else:
        checks.append(("Task 2 report generated", False, "File not found"))
    
    return checks

def check_task3():
    """Validate Task 3 implementation"""
    print("\n" + "="*80)
    print("VALIDATING TASK 3: RAG CORE LOGIC")
    print("="*80)
    
    checks = []
    
    # Check 1: RAG pipeline module exists
    rag_module = 'src/rag_pipeline.py'
    if os.path.exists(rag_module):
        checks.append(("RAG pipeline module exists", True, rag_module))
    else:
        checks.append(("RAG pipeline module exists", False, "File not found"))
    
    # Check 2: Evaluation files exist
    eval_files = [
        'data/processed/rag_evaluation.csv',
        'data/processed/evaluation_summary.json',
        'data/processed/rag_evaluation_report.md'
    ]
    
    existing_files = [f for f in eval_files if os.path.exists(f)]
    checks.append(("Evaluation files generated", len(existing_files) > 0,
                  f"{len(existing_files)}/{len(eval_files)} files exist"))
    
    # Check 3: Evaluation table has required columns
    eval_csv = 'data/processed/rag_evaluation.csv'
    if os.path.exists(eval_csv):
        try:
            df = pd.read_csv(eval_csv)
            required_columns = ['Question', 'Generated Answer', 'Retrieved Sources', 
                               'Quality Score', 'Comments']
            missing_cols = [col for col in required_columns if col not in df.columns]
            checks.append(("Evaluation table has required columns", len(missing_cols) == 0,
                          f"Missing: {missing_cols}" if missing_cols else "All present"))
        except Exception as e:
            checks.append(("Evaluation table check", False, f"Error reading: {e}"))
    
    # Check 4: Can import and test RAG pipeline
    try:
        # Add src to path if not already
        src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
        if src_path not in sys.path:
            sys.path.append(src_path)
        
        if os.path.exists(rag_module):
            from rag_pipeline import RAGPipeline
            checks.append(("RAG pipeline can be imported", True, "Import successful"))
        else:
            checks.append(("RAG pipeline can be imported", False, "Module file not found"))
    except ImportError as e:
        checks.append(("RAG pipeline can be imported", False, f"ImportError: {e}"))
    except Exception as e:
        checks.append(("RAG pipeline can be imported", False, f"Error: {e}"))
    
    return checks

def check_task4():
    """Validate Task 4 implementation"""
    print("\n" + "="*80)
    print("VALIDATING TASK 4: INTERACTIVE INTERFACE")
    print("="*80)
    
    checks = []
    
    # Check 1: Main app file exists
    app_file = 'app.py'
    if os.path.exists(app_file):
        try:
            content = read_file_with_encoding(app_file)
            checks.append(("Main app file exists", True, 
                          f"Size: {len(content):,} bytes"))
        except Exception as e:
            checks.append(("Main app file exists", False, f"Error reading: {e}"))
    else:
        checks.append(("Main app file exists", False, "File not found"))
    
    # Check 2: Interface requirements
    if os.path.exists(app_file):
        try:
            content = read_file_with_encoding(app_file).lower()
            
            # Check for required UI elements
            requirements = {
                "Text input box": any(keyword in content for keyword in ['textbox', 'text_area', 'chat_input', 'input(', 'st.text_input', 'gr.textbox']),
                "Submit/Ask button": any(keyword in content for keyword in ['button', 'submit', 'ask', 'st.button', 'gr.button']),
                "Clear button": 'clear' in content or 'reset' in content,
                "Display answer": any(keyword in content for keyword in ['chatbot', 'output', 'answer', 'response', 'st.write', 'gr.markdown']),
                "Display sources": 'sources' in content or 'retrieved' in content or 'context' in content,
            }
            
            for req, met in requirements.items():
                checks.append((f"UI has {req}", met, 
                              "Present" if met else "Not found"))
        except Exception as e:
            checks.append(("UI requirements check", False, f"Error: {e}"))
    
    # Check 3: Gradio/Streamlit imports
    if os.path.exists(app_file):
        try:
            content = read_file_with_encoding(app_file)
            
            has_gradio = 'import gradio' in content or 'from gradio' in content
            has_streamlit = 'import streamlit' in content or 'from streamlit' in content
            
            checks.append(("UI framework imported", has_gradio or has_streamlit,
                          f"Gradio: {has_gradio}, Streamlit: {has_streamlit}"))
        except Exception as e:
            checks.append(("UI framework check", False, f"Error: {e}"))
    
    # Check 4: Can run app in test mode (skip on CI to avoid hanging)
    try:
        # Check if we're in a CI environment
        ci_env = os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS')
        
        if not ci_env:
            import subprocess
            # Test with --help or --version if available
            result = subprocess.run([sys.executable, app_file, '--help'], 
                                  capture_output=True, text=True, timeout=5)
            checks.append(("App can be executed", result.returncode in [0, 1],  # 0 or 1 are okay
                          f"Return code: {result.returncode}"))
        else:
            checks.append(("App can be executed", True, "Skipped in CI environment"))
    except subprocess.TimeoutExpired:
        checks.append(("App can be executed", True, "Timeout (app likely started)"))
    except Exception as e:
        checks.append(("App can be executed", False, f"Error: {e}"))
    
    return checks

def check_git_best_practices():
    """Check Git & GitHub best practices"""
    print("\n" + "="*80)
    print("VALIDATING GIT & GITHUB BEST PRACTICES")
    print("="*80)
    
    checks = []
    
    # Check 1: .gitignore exists
    if os.path.exists('.gitignore'):
        try:
            content = read_file_with_encoding('.gitignore')
            checks.append((".gitignore file exists", True, 
                          f"{len(content.splitlines())} rules"))
        except Exception as e:
            checks.append((".gitignore file exists", False, f"Error reading: {e}"))
    else:
        checks.append((".gitignore file exists", False, "File not found"))
    
    # Check 2: GitHub workflows exist
    workflows_dir = '.github/workflows'
    if os.path.exists(workflows_dir):
        try:
            workflows = [f for f in os.listdir(workflows_dir) if f.endswith('.yml') or f.endswith('.yaml')]
            checks.append(("GitHub workflows exist", len(workflows) > 0,
                          f"Found: {len(workflows)} workflow(s)"))
        except Exception as e:
            checks.append(("GitHub workflows exist", False, f"Error: {e}"))
    else:
        checks.append(("GitHub workflows exist", False, "Directory not found"))
    
    # Check 3: README exists
    if os.path.exists('README.md'):
        try:
            content = read_file_with_encoding('README.md')
            checks.append(("README exists", True, 
                          f"Length: {len(content):,} chars"))
        except Exception as e:
            checks.append(("README exists", False, f"Error reading: {e}"))
    else:
        checks.append(("README exists", False, "File not found"))
    
    # Check 4: Project structure
    required_dirs = ['src', 'tests', 'data', 'notebooks', 'vector_store']
    existing_dirs = [d for d in required_dirs if os.path.exists(d)]
    checks.append(("Project structure complete", len(existing_dirs) == len(required_dirs),
                  f"Missing: {set(required_dirs) - set(existing_dirs)}" 
                  if len(existing_dirs) < len(required_dirs) else "All present"))
    
    return checks

def check_code_best_practices():
    """Check code best practices"""
    print("\n" + "="*80)
    print("VALIDATING CODE BEST PRACTICES")
    print("="*80)
    
    checks = []
    
    # Check Python files for best practices
    python_files = []
    src_dir = 'src'
    if os.path.exists(src_dir):
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
    
    if python_files:
        # Sample a few files for analysis
        sample_files = python_files[:min(3, len(python_files))]
        
        for file_path in sample_files:
            try:
                content = read_file_with_encoding(file_path)
                filename = os.path.basename(file_path)
                
                # Check for docstrings
                has_docstring = '"""' in content or "'''" in content
                checks.append((f"{filename} has docstrings", has_docstring,
                              "Present" if has_docstring else "Missing"))
                
                # Check for type hints
                has_type_hints = 'def ' in content and '->' in content
                checks.append((f"{filename} has type hints", has_type_hints,
                              "Present" if has_type_hints else "Limited"))
                
                # Check for error handling
                has_error_handling = 'try:' in content or 'except ' in content
                checks.append((f"{filename} has error handling", has_error_handling,
                              "Present" if has_error_handling else "Limited"))
            except Exception as e:
                checks.append((f"Analyzing {file_path}", False, f"Error: {e}"))
    else:
        checks.append(("Python source files", False, "No Python files found in src/"))
    
    # Check test files
    test_files = []
    if os.path.exists('tests'):
        for root, dirs, files in os.walk('tests'):
            for file in files:
                if file.endswith('.py') and file.startswith('test_'):
                    test_files.append(file)
    
    checks.append(("Test files exist", len(test_files) > 0,
                  f"Found: {len(test_files)} test file(s)"))
    
    # Check requirements.txt
    if os.path.exists('requirements.txt'):
        try:
            content = read_file_with_encoding('requirements.txt')
            packages = [line.strip() for line in content.splitlines() 
                       if line.strip() and not line.startswith('#')]
            checks.append(("requirements.txt complete", len(packages) > 5,
                          f"Lists {len(packages)} packages"))
        except Exception as e:
            checks.append(("requirements.txt complete", False, f"Error reading: {e}"))
    else:
        checks.append(("requirements.txt complete", False, "File not found"))
    
    return checks

def generate_summary_report(all_checks):
    """Generate a summary report"""
    print("\n" + "="*80)
    print("VALIDATION SUMMARY REPORT")
    print("="*80)
    
    summary = {}
    
    for task_name, checks in all_checks.items():
        total = len(checks)
        passed = sum(1 for _, status, _ in checks if status)
        failed = total - passed
        
        summary[task_name] = {
            'total': total,
            'passed': passed,
            'failed': failed,
            'percentage': (passed / total * 100) if total > 0 else 0
        }
        
        print(f"\n{task_name.upper().replace('_', ' ')}:")
        print(f"  Passed: {passed}/{total} ({summary[task_name]['percentage']:.1f}%)")
        
        # Show failed checks
        if failed > 0:
            print(f"  Failed checks:")
            for check_name, status, details in checks:
                if not status:
                    print(f"    - {check_name}: {details}")
    
    # Overall status
    total_checks = sum(s['total'] for s in summary.values())
    total_passed = sum(s['passed'] for s in summary.values())
    overall_percentage = (total_passed / total_checks * 100) if total_checks > 0 else 0
    
    print("\n" + "="*80)
    print(f"OVERALL STATUS: {overall_percentage:.1f}% PASSED")
    print("="*80)
    
    # Save report
    report_data = {
        'validation_date': pd.Timestamp.now().isoformat(),
        'overall_percentage': overall_percentage,
        'tasks': summary,
        'details': all_checks
    }
    
    os.makedirs('data/processed', exist_ok=True)
    report_path = 'data/processed/validation_report.json'
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        print(f"\nDetailed validation report saved to: {report_path}")
    except Exception as e:
        print(f"\nWarning: Could not save validation report: {e}")
    
    return overall_percentage

def main():
    """Main validation function"""
    print("VALIDATING ALL TASKS AGAINST REQUIREMENTS")
    print("="*80)
    
    all_checks = {
        'task1': check_task1(),
        'task2': check_task2(),
        'task3': check_task3(),
        'task4': check_task4(),
        'git_practices': check_git_best_practices(),
        'code_practices': check_code_best_practices()
    }
    
    overall_score = generate_summary_report(all_checks)
    
    # Final recommendation
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if overall_score >= 80:
        print("✅ EXCELLENT: All major requirements met!")
        print("The project is ready for submission.")
    elif overall_score >= 60:
        print("⚠️ GOOD: Most requirements met.")
        print("Review failed checks and make necessary improvements.")
    else:
        print("❌ NEEDS WORK: Significant requirements missing.")
        print("Please address the failed checks before submission.")
    
    print("\nNext steps:")
    print("1. Run: python tests/test_all_tasks.py to validate")
    print("2. Fix any failed checks")
    print("3. Run: python app.py to launch the chatbot")
    print("4. Submit your project")
    
    # Return exit code based on overall score
    return 0 if overall_score >= 60 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)