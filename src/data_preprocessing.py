# src/data_preprocessing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import re
import os
from typing import Tuple, List, Dict
import json
from pathlib import Path
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import textstat

warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class ComplaintDataPreprocessor:
    """Class for preprocessing CFPB complaint data"""
    
    def __init__(self, data_path: str = None):
        """
        Initialize the preprocessor
        
        Args:
            data_path: Path to the CFPB complaint dataset
        """
        self.data_path = data_path
        self.df = None
        self.filtered_df = None
        self.target_products = [
            'Credit card', 
            'Personal loan', 
            'Savings account', 
            'Money transfers'
        ]
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the CFPB complaint dataset
        
        Returns:
            DataFrame containing the complaint data
        """
        if self.data_path and os.path.exists(self.data_path):
            print(f"Loading data from {self.data_path}")
            # Try different encodings for CSV files
            encodings = ['utf-8', 'latin-1', 'ISO-8859-1']
            
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.data_path, encoding=encoding, low_memory=False)
                    print(f"Successfully loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Error with {encoding}: {e}")
        else:
            # Try to load from common CFPB dataset URLs or sample data
            print("No local data found. Please download the CFPB complaint dataset.")
            print("You can download it from: https://www.consumerfinance.gov/data-research/consumer-complaints/#download-the-data")
            return None
        
        print(f"Dataset loaded with shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        return self.df
    
    def perform_initial_eda(self) -> Dict:
        """
        Perform initial exploratory data analysis
        
        Returns:
            Dictionary containing EDA statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        eda_stats = {}
        
        # Basic statistics
        eda_stats['total_records'] = len(self.df)
        eda_stats['total_columns'] = len(self.df.columns)
        eda_stats['missing_values'] = self.df.isnull().sum().sum()
        eda_stats['missing_percentage'] = (eda_stats['missing_values'] / (self.df.shape[0] * self.df.shape[1])) * 100
        
        # Data types
        eda_stats['data_types'] = self.df.dtypes.value_counts().to_dict()
        
        # Check for narrative column
        narrative_cols = [col for col in self.df.columns if 'narrative' in col.lower() or 'complaint' in col.lower()]
        eda_stats['narrative_columns'] = narrative_cols
        
        # Identify key columns
        possible_product_cols = [col for col in self.df.columns if 'product' in col.lower()]
        possible_issue_cols = [col for col in self.df.columns if 'issue' in col.lower()]
        
        eda_stats['possible_product_cols'] = possible_product_cols
        eda_stats['possible_issue_cols'] = possible_issue_cols
        
        print("\n=== Initial EDA Summary ===")
        print(f"Total records: {eda_stats['total_records']:,}")
        print(f"Total columns: {eda_stats['total_columns']}")
        print(f"Missing values: {eda_stats['missing_values']:,} ({eda_stats['missing_percentage']:.2f}%)")
        print(f"Narrative columns found: {eda_stats['narrative_columns']}")
        print(f"Possible product columns: {eda_stats['possible_product_cols']}")
        print(f"Possible issue columns: {eda_stats['possible_issue_cols']}")
        
        return eda_stats
    
    def analyze_product_distribution(self) -> pd.DataFrame:
        """
        Analyze distribution of complaints across different products
        
        Returns:
            DataFrame with product distribution statistics
        """
        # Find the actual product column
        product_col = None
        for col in self.df.columns:
            if 'product' in col.lower():
                product_col = col
                break
        
        if product_col is None:
            print("No product column found. Checking column names:")
            print(self.df.columns.tolist())
            return None
        
        print(f"\n=== Product Distribution Analysis ===")
        print(f"Using product column: '{product_col}'")
        
        # Get product distribution
        product_dist = self.df[product_col].value_counts()
        
        print(f"\nTotal unique products: {len(product_dist)}")
        print(f"\nTop 10 products by complaint count:")
        for i, (product, count) in enumerate(product_dist.head(10).items()):
            print(f"{i+1}. {product}: {count:,} complaints ({count/len(self.df)*100:.1f}%)")
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        top_20 = product_dist.head(20)
        bars = plt.barh(range(len(top_20)), top_20.values)
        plt.yticks(range(len(top_20)), top_20.index)
        plt.xlabel('Number of Complaints')
        plt.title('Top 20 Products by Complaint Count')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_20.values)):
            plt.text(value + 100, i, f'{value:,}', va='center')
        
        plt.tight_layout()
        plt.savefig('data/processed/product_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return product_dist
    
    def analyze_narrative_lengths(self) -> Dict:
        """
        Analyze length distribution of complaint narratives
        
        Returns:
            Dictionary containing narrative length statistics
        """
        # Find narrative column
        narrative_col = None
        for col in self.df.columns:
            if any(term in col.lower() for term in ['narrative', 'complaint', 'description']):
                # Check if this column contains text data
                if self.df[col].dtype == 'object' and self.df[col].str.len().mean() > 10:
                    narrative_col = col
                    break
        
        if narrative_col is None:
            print("No narrative column found.")
            return None
        
        print(f"\n=== Narrative Length Analysis ===")
        print(f"Using narrative column: '{narrative_col}'")
        
        # Calculate narrative lengths
        self.df['narrative_length'] = self.df[narrative_col].astype(str).apply(
            lambda x: len(x.split()) if pd.notnull(x) and x.strip() != '' and x.strip().lower() != 'nan' else 0
        )
        
        # Filter out zero-length narratives
        valid_narratives = self.df[self.df['narrative_length'] > 0]
        
        length_stats = {
            'total_with_narrative': len(valid_narratives),
            'total_without_narrative': len(self.df) - len(valid_narratives),
            'percent_with_narrative': (len(valid_narratives) / len(self.df)) * 100,
            'mean_length': valid_narratives['narrative_length'].mean(),
            'median_length': valid_narratives['narrative_length'].median(),
            'std_length': valid_narratives['narrative_length'].std(),
            'min_length': valid_narratives['narrative_length'].min(),
            'max_length': valid_narratives['narrative_length'].max(),
            'q1_length': valid_narratives['narrative_length'].quantile(0.25),
            'q3_length': valid_narratives['narrative_length'].quantile(0.75)
        }
        
        print(f"\nNarrative Statistics:")
        print(f"Complaints with narrative: {length_stats['total_with_narrative']:,} ({length_stats['percent_with_narrative']:.1f}%)")
        print(f"Complaints without narrative: {length_stats['total_without_narrative']:,}")
        print(f"Mean narrative length: {length_stats['mean_length']:.1f} words")
        print(f"Median narrative length: {length_stats['median_length']:.1f} words")
        print(f"Std narrative length: {length_stats['std_length']:.1f} words")
        print(f"Min length: {length_stats['min_length']} words")
        print(f"Max length: {length_stats['max_length']} words")
        print(f"25th percentile: {length_stats['q1_length']:.1f} words")
        print(f"75th percentile: {length_stats['q3_length']:.1f} words")
        
        # Create visualizations
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram of narrative lengths
        axes[0].hist(valid_narratives['narrative_length'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Narrative Length (words)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Narrative Lengths')
        axes[0].axvline(length_stats['mean_length'], color='red', linestyle='--', label=f'Mean: {length_stats["mean_length"]:.1f}')
        axes[0].axvline(length_stats['median_length'], color='green', linestyle='--', label=f'Median: {length_stats["median_length"]:.1f}')
        axes[0].legend()
        
        # Box plot
        axes[1].boxplot(valid_narratives['narrative_length'], vert=False)
        axes[1].set_xlabel('Narrative Length (words)')
        axes[1].set_title('Box Plot of Narrative Lengths')
        
        plt.tight_layout()
        plt.savefig('data/processed/narrative_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyze very short and very long narratives
        short_threshold = 10  # Very short narratives
        long_threshold = 500  # Very long narratives
        
        very_short = valid_narratives[valid_narratives['narrative_length'] <= short_threshold]
        very_long = valid_narratives[valid_narratives['narrative_length'] >= long_threshold]
        
        print(f"\nVery short narratives (≤ {short_threshold} words): {len(very_short):,} ({len(very_short)/len(valid_narratives)*100:.1f}%)")
        print(f"Very long narratives (≥ {long_threshold} words): {len(very_long):,} ({len(very_long)/len(valid_narratives)*100:.1f}%)")
        
        if len(very_short) > 0:
            print("\nSample of very short narratives:")
            for i, (_, row) in enumerate(very_short.head(3).iterrows()):
                print(f"{i+1}. Length: {row['narrative_length']} words")
                print(f"   Preview: {row[narrative_col][:100]}...")
        
        length_stats['very_short_count'] = len(very_short)
        length_stats['very_long_count'] = len(very_long)
        
        return length_stats
    
    def filter_dataset(self) -> pd.DataFrame:
        """
        Filter dataset to meet project requirements
        
        Returns:
            Filtered DataFrame
        """
        print("\n=== Filtering Dataset ===")
        
        # Find product column
        product_col = None
        for col in self.df.columns:
            if 'product' in col.lower():
                product_col = col
                break
        
        # Find narrative column
        narrative_col = None
        for col in self.df.columns:
            if any(term in col.lower() for term in ['narrative', 'complaint', 'description']):
                if self.df[col].dtype == 'object':
                    narrative_col = col
                    break
        
        if product_col is None or narrative_col is None:
            print(f"Required columns not found. Product: {product_col}, Narrative: {narrative_col}")
            return None
        
        print(f"Product column: {product_col}")
        print(f"Narrative column: {narrative_col}")
        
        # Step 1: Filter for target products
        print(f"\nFiltering for target products: {self.target_products}")
        
        # Handle case-insensitive matching and variations
        filtered_df = self.df.copy()
        
        # Create a mask for target products (case-insensitive)
        product_mask = filtered_df[product_col].astype(str).str.lower().isin(
            [p.lower() for p in self.target_products]
        )
        
        # Also check for partial matches
        for product in self.target_products:
            partial_mask = filtered_df[product_col].astype(str).str.lower().str.contains(product.lower())
            product_mask = product_mask | partial_mask
        
        filtered_df = filtered_df[product_mask].copy()
        print(f"After product filtering: {len(filtered_df):,} complaints")
        
        # Step 2: Remove empty narratives
        print(f"\nRemoving empty narratives...")
        
        # Create function to check if narrative is valid
        def is_valid_narrative(text):
            if pd.isna(text):
                return False
            text_str = str(text).strip()
            if text_str == '' or text_str.lower() in ['nan', 'null', 'none', 'na']:
                return False
            # Check if it has meaningful content (more than just whitespace/punctuation)
            if len(re.findall(r'\w+', text_str)) < 3:  # At least 3 words
                return False
            return True
        
        narrative_mask = filtered_df[narrative_col].apply(is_valid_narrative)
        filtered_df = filtered_df[narrative_mask].copy()
        
        print(f"After removing empty narratives: {len(filtered_df):,} complaints")
        
        # Step 3: Standardize product names
        print(f"\nStandardizing product names...")
        
        # Map variations to standard product names
        product_mapping = {}
        for target_product in self.target_products:
            # Find variations of this product
            variations = filtered_df[filtered_df[product_col].astype(str).str.lower().str.contains(
                target_product.lower()
            )][product_col].unique()
            
            for var in variations:
                product_mapping[var] = target_product
        
        filtered_df[product_col] = filtered_df[product_col].map(product_mapping).fillna(filtered_df[product_col])
        
        # Final product distribution
        print(f"\nFinal product distribution:")
        final_dist = filtered_df[product_col].value_counts()
        for product, count in final_dist.items():
            print(f"  {product}: {count:,} complaints ({count/len(filtered_df)*100:.1f}%)")
        
        self.filtered_df = filtered_df
        return filtered_df
    
    def clean_text_narratives(self) -> pd.DataFrame:
        """
        Clean text narratives to improve embedding quality
        
        Returns:
            DataFrame with cleaned narratives
        """
        if self.filtered_df is None:
            raise ValueError("Filtered data not available. Call filter_dataset() first.")
        
        print("\n=== Cleaning Text Narratives ===")
        
        # Find narrative column
        narrative_col = None
        for col in self.filtered_df.columns:
            if any(term in col.lower() for term in ['narrative', 'complaint', 'description']):
                if self.filtered_df[col].dtype == 'object':
                    narrative_col = col
                    break
        
        if narrative_col is None:
            print("No narrative column found in filtered data")
            return self.filtered_df
        
        # Create a copy for cleaning
        cleaned_df = self.filtered_df.copy()
        
        # Create new column for cleaned narratives
        cleaned_col = 'cleaned_narrative'
        
        # Define cleaning functions
        def clean_text(text):
            if pd.isna(text):
                return ""
            
            text_str = str(text)
            
            # 1. Lowercase
            text_str = text_str.lower()
            
            # 2. Remove common boilerplate text
            boilerplate_patterns = [
                r'i am writing to file a complaint',
                r'dear sir or madam',
                r'to whom it may concern',
                r'this is a complaint regarding',
                r'i am writing this letter to complain',
                r'please be advised that',
                r'i would like to file a formal complaint',
                r'this complaint is about',
                r'i am writing to express my dissatisfaction',
                r'regarding my complaint about',
                r'cfpb complaint',
                r'consumer financial protection bureau',
                r'complaint id:?\s*\d+',
                r'date:?\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}',
                r'account number:?\s*[\w\*]+',
                r'social security number:?\s*[\d\*]+',
                r'phone number:?\s*[\d\-\(\)\s]+',
                r'email:?\s*[\w\.\-@]+',
                r'address:?\s*[\w\s\.,]+',
                r'confidential - attorney client privileged',
                r'privileged and confidential',
                r'xxxx',  # Redacted information
                r'\*{4,}',  # Multiple asterisks
            ]
            
            for pattern in boilerplate_patterns:
                text_str = re.sub(pattern, '', text_str, flags=re.IGNORECASE)
            
            # 3. Remove special characters (keep basic punctuation for sentence structure)
            # Keep: letters, numbers, basic punctuation (.!?,;:)
            text_str = re.sub(r'[^a-zA-Z0-9\s\.!?,;:\-"\']', ' ', text_str)
            
            # 4. Remove extra whitespace
            text_str = re.sub(r'\s+', ' ', text_str).strip()
            
            # 5. Remove very short sentences (likely artifacts)
            sentences = re.split(r'[.!?]+', text_str)
            valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
            text_str = '. '.join(valid_sentences)
            
            return text_str
        
        print("Cleaning narratives...")
        cleaned_df[cleaned_col] = cleaned_df[narrative_col].apply(clean_text)
        
        # Calculate cleaning statistics
        original_lengths = cleaned_df[narrative_col].astype(str).apply(
            lambda x: len(x.split()) if pd.notnull(x) else 0
        )
        cleaned_lengths = cleaned_df[cleaned_col].apply(
            lambda x: len(x.split()) if pd.notnull(x) else 0
        )
        
        print(f"\nCleaning Statistics:")
        print(f"Average original length: {original_lengths.mean():.1f} words")
        print(f"Average cleaned length: {cleaned_lengths.mean():.1f} words")
        print(f"Average reduction: {((original_lengths - cleaned_lengths) / original_lengths * 100).mean():.1f}%")
        
        # Show samples
        print(f"\nSample cleaning results:")
        sample_idx = cleaned_df.head(3).index
        for idx in sample_idx:
            original = str(cleaned_df.loc[idx, narrative_col])[:150]
            cleaned = str(cleaned_df.loc[idx, cleaned_col])[:150]
            print(f"\nOriginal (first 150 chars): {original}...")
            print(f"Cleaned (first 150 chars): {cleaned}...")
            print("-" * 50)
        
        self.filtered_df = cleaned_df
        return cleaned_df
    
    def save_filtered_data(self, output_path: str = 'data/processed/filtered_complaints.csv'):
        """
        Save the filtered and cleaned dataset
        
        Args:
            output_path: Path to save the filtered data
        """
        if self.filtered_df is None:
            raise ValueError("No filtered data to save")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        self.filtered_df.to_csv(output_path, index=False)
        print(f"\nFiltered dataset saved to: {output_path}")
        print(f"Shape: {self.filtered_df.shape}")
        
        return output_path
    
    def generate_eda_report(self, output_path: str = 'data/processed/eda_report.txt'):
        """
        Generate a comprehensive EDA report
        
        Args:
            output_path: Path to save the report
        """
        report_lines = []
        
        report_lines.append("=" * 60)
        report_lines.append("EXPLORATORY DATA ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append("\n")
        
        # Basic dataset info
        report_lines.append("1. DATASET OVERVIEW")
        report_lines.append("-" * 40)
        report_lines.append(f"Total records: {len(self.df):,}")
        report_lines.append(f"Total columns: {len(self.df.columns)}")
        report_lines.append(f"Missing values: {self.df.isnull().sum().sum():,}")
        report_lines.append(f"Missing percentage: {(self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1]) * 100):.2f}%")
        
        # Narrative analysis
        if 'narrative_length' in self.df.columns:
            valid_narratives = self.df[self.df['narrative_length'] > 0]
            report_lines.append(f"\nComplaints with narrative: {len(valid_narratives):,} ({(len(valid_narratives) / len(self.df)) * 100:.1f}%)")
            report_lines.append(f"Complaints without narrative: {len(self.df) - len(valid_narratives):,}")
            report_lines.append(f"Average narrative length: {valid_narratives['narrative_length'].mean():.1f} words")
            report_lines.append(f"Median narrative length: {valid_narratives['narrative_length'].median():.1f} words")
        
        # Filtered dataset info
        if self.filtered_df is not None:
            report_lines.append("\n\n2. FILTERED DATASET")
            report_lines.append("-" * 40)
            report_lines.append(f"Total filtered complaints: {len(self.filtered_df):,}")
            
            # Product distribution in filtered data
            product_col = None
            for col in self.filtered_df.columns:
                if 'product' in col.lower():
                    product_col = col
                    break
            
            if product_col:
                report_lines.append("\nProduct distribution in filtered data:")
                product_dist = self.filtered_df[product_col].value_counts()
                for product, count in product_dist.items():
                    percentage = (count / len(self.filtered_df)) * 100
                    report_lines.append(f"  - {product}: {count:,} complaints ({percentage:.1f}%)")
            
            # Narrative length in filtered data
            if 'narrative_length' in self.filtered_df.columns:
                report_lines.append(f"\nAverage narrative length in filtered data: {self.filtered_df['narrative_length'].mean():.1f} words")
        
        # Key findings
        report_lines.append("\n\n3. KEY FINDINGS")
        report_lines.append("-" * 40)
        report_lines.append("1. The dataset contains a significant number of complaints across various financial products.")
        report_lines.append("2. Not all complaints contain detailed narratives - some are very brief or missing entirely.")
        report_lines.append("3. Narrative lengths vary widely, with some being very short and others very long.")
        report_lines.append("4. Common boilerplate text and personal information are present in many narratives.")
        report_lines.append("5. After filtering for target products and cleaning, we have a focused dataset suitable for RAG.")
        
        # Recommendations
        report_lines.append("\n\n4. RECOMMENDATIONS FOR RAG PIPELINE")
        report_lines.append("-" * 40)
        report_lines.append("1. Implement text chunking to handle varying narrative lengths.")
        report_lines.append("2. Use metadata (product type, issue, etc.) to improve retrieval relevance.")
        report_lines.append("3. Consider stratified sampling to ensure representation across all product categories.")
        report_lines.append("4. Apply additional text normalization if needed (lemmatization, stopword removal).")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\nEDA report saved to: {output_path}")
        
        return '\n'.join(report_lines)

def main():
    """Main execution function"""
    # Initialize preprocessor
    preprocessor = ComplaintDataPreprocessor()
    
    # Load data (update path to your actual data file)
    data_path = input("Enter path to CFPB complaint dataset (or press Enter to use default): ").strip()
    if not data_path:
        # Try to find the data in common locations
        possible_paths = [
            'data/raw/complaints.csv',
            'data/raw/consumer_complaints.csv',
            'complaints.csv',
            'consumer_complaints.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if not data_path:
            print("No data file found. Please download the CFPB complaint dataset.")
            print("You can download it from: https://www.consumerfinance.gov/data-research/consumer-complaints/#download-the-data")
            return
    
    preprocessor.data_path = data_path
    df = preprocessor.load_data()
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Perform EDA
    eda_stats = preprocessor.perform_initial_eda()
    product_dist = preprocessor.analyze_product_distribution()
    length_stats = preprocessor.analyze_narrative_lengths()
    
    # Filter dataset
    filtered_df = preprocessor.filter_dataset()
    
    if filtered_df is not None:
        # Clean narratives
        cleaned_df = preprocessor.clean_text_narratives()
        
        # Save filtered data
        output_path = preprocessor.save_filtered_data()
        
        # Generate EDA report
        report = preprocessor.generate_eda_report()
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"\nFinal dataset shape: {cleaned_df.shape}")
        print(f"Saved to: {output_path}")
        print("\nYou can now proceed to Task 2: Text Chunking and Embedding.")

if __name__ == "__main__":
    main()