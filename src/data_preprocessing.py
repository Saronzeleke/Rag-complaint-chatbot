# src/data_preprocessing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import os
from typing import Dict
import nltk
from pathlib import Path

warnings.filterwarnings('ignore')
# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class ComplaintDataPreprocessor:
    """Class for preprocessing complaint ticket data"""
    
    def __init__(self, data_path: str = None):
        """
        Initialize the preprocessor
        
        Args:
            data_path: Path to the complaint dataset
        """
        self.data_path = data_path or r'C:\Users\admin\Rag-complaint-chatbot\data\raw\complaints.csv'
        self.df = None
        self.filtered_df = None
        self.target_products = [
            'Credit card', 
            'Personal loan', 
            'Savings account', 
            'Money transfers'
        ]
        self.product_column = None
        self.narrative_column = None
        
    def _identify_columns(self):
        """Identify key columns in the dataset"""
        if self.df is None:
            return
            
        print("\n=== Column Identification ===")
        print(f"Dataset columns: {list(self.df.columns)}")
        
        # Look for product/service related columns
        product_keywords = ['product', 'service', 'category', 'type', 'department']
        narrative_keywords = ['description', 'narrative', 'complaint', 'issue', 'details', 'comment', 'text']
        
        for col in self.df.columns:
            col_lower = col.lower()
            
            # Check for product/service column
            if any(keyword in col_lower for keyword in product_keywords):
                if self.product_column is None:
                    self.product_column = col
                    print(f"Identified product column: '{col}'")
            
            # Check for narrative/description column
            if any(keyword in col_lower for keyword in narrative_keywords):
                if self.df[col].dtype == 'object' and self.df[col].notna().any():
                    if self.narrative_column is None:
                        self.narrative_column = col
                        print(f"Identified narrative column: '{col}'")
        
        # If not found, try to guess based on data
        if self.product_column is None:
            for col in self.df.columns:
                if self.df[col].dtype == 'object' and self.df[col].nunique() < 100:
                    self.product_column = col
                    print(f"Guessed product column: '{col}' (low cardinality text)")
                    break
        
        if self.narrative_column is None:
            # Find the column with longest text
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    avg_len = self.df[col].astype(str).str.len().mean()
                    if avg_len > 50:  # Reasonable text length
                        self.narrative_column = col
                        print(f"Guessed narrative column: '{col}' (average length: {avg_len:.0f} chars)")
                        break
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the complaint dataset
        
        Returns:
            DataFrame containing the complaint data
        """
        print(f"Loading data from {self.data_path}")
        
        # Check if file exists
        if not os.path.exists(self.data_path):
            print(f"Error: File not found at {self.data_path}")
            print("Please ensure the file exists at the specified path.")
            return None
        
        # Try different encodings and delimiters
        encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                # Try CSV first
                self.df = pd.read_csv(self.data_path, encoding=encoding, low_memory=False, on_bad_lines='skip')
                print(f"Successfully loaded CSV with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error with {encoding}: {e}")
                # Try with different delimiters
                try:
                    self.df = pd.read_csv(self.data_path, encoding=encoding, sep=';', low_memory=False, on_bad_lines='skip')
                    print(f"Successfully loaded with semicolon delimiter and {encoding} encoding")
                    break
                except:
                    try:
                        self.df = pd.read_csv(self.data_path, encoding=encoding, sep='\t', low_memory=False, on_bad_lines='skip')
                        print(f"Successfully loaded with tab delimiter and {encoding} encoding")
                        break
                    except:
                        continue
        
        if self.df is None:
            print("Failed to load the file. Please check the file format.")
            return None
        
        print(f"Dataset loaded with shape: {self.df.shape}")
        print(f"First few rows:")
        print(self.df.head())
        
        # Identify key columns
        self._identify_columns()
        
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
        
        # Column-specific analysis
        if self.product_column:
            eda_stats['unique_products'] = self.df[self.product_column].nunique()
            eda_stats['top_products'] = self.df[self.product_column].value_counts().head(10).to_dict()
        
        if self.narrative_column:
            # Count non-empty narratives
            non_empty = self.df[self.narrative_column].notna() & (self.df[self.narrative_column].astype(str).str.strip() != '')
            eda_stats['non_empty_narratives'] = non_empty.sum()
            eda_stats['empty_narratives'] = len(self.df) - non_empty.sum()
        
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        print(f"\nDataset Overview:")
        print(f"Total records: {eda_stats['total_records']:,}")
        print(f"Total columns: {eda_stats['total_columns']}")
        print(f"Missing values: {eda_stats['missing_values']:,} ({eda_stats['missing_percentage']:.2f}%)")
        
        if self.product_column:
            print(f"\nProduct Analysis:")
            print(f"Product column: '{self.product_column}'")
            print(f"Unique products/services: {eda_stats['unique_products']}")
            print(f"\nTop 10 products/services:")
            for product, count in eda_stats['top_products'].items():
                print(f"  {product}: {count:,} ({count/len(self.df)*100:.1f}%)")
        
        if self.narrative_column:
            print(f"\nNarrative Analysis:")
            print(f"Narrative column: '{self.narrative_column}'")
            print(f"Non-empty narratives: {eda_stats['non_empty_narratives']:,} ({eda_stats['non_empty_narratives']/len(self.df)*100:.1f}%)")
            print(f"Empty narratives: {eda_stats['empty_narratives']:,} ({eda_stats['empty_narratives']/len(self.df)*100:.1f}%)")
        
        return eda_stats
    
    def analyze_product_distribution(self):
        """Analyze distribution of complaints across different products/services"""
        if self.df is None or self.product_column is None:
            print("No product data available for analysis")
            return None
        
        print("\n" + "="*60)
        print("PRODUCT/SERVICE DISTRIBUTION ANALYSIS")
        print("="*60)
        
        product_dist = self.df[self.product_column].value_counts()
        
        print(f"\nTotal unique categories: {len(product_dist)}")
        print(f"\nDistribution of complaints:")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart for top 20
        top_n = min(20, len(product_dist))
        top_products = product_dist.head(top_n)
        
        axes[0].barh(range(len(top_products)), top_products.values)
        axes[0].set_yticks(range(len(top_products)))
        axes[0].set_yticklabels(top_products.index)
        axes[0].set_xlabel('Number of Complaints')
        axes[0].set_title(f'Top {top_n} Products/Services by Complaint Count')
        axes[0].invert_yaxis()
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(axes[0].patches, top_products.values)):
            axes[0].text(value + max(value * 0.01, 1), i, f'{value:,}', va='center')
        
        # Pie chart for top 10
        top_10 = product_dist.head(10)
        other = product_dist[10:].sum() if len(product_dist) > 10 else 0
        
        if other > 0:
            top_10 = pd.concat([top_10, pd.Series({'Other': other})])
        
        axes[1].pie(top_10.values, labels=top_10.index, autopct='%1.1f%%')
        axes[1].set_title('Complaint Distribution (Top 10 + Others)')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('data/processed', exist_ok=True)
        plt.savefig('data/processed/product_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return product_dist
    
    def analyze_narrative_lengths(self) -> Dict:
        """Analyze length distribution of complaint narratives"""
        if self.df is None or self.narrative_column is None:
            print("No narrative data available for analysis")
            return None
        
        print("\n" + "="*60)
        print("NARRATIVE LENGTH ANALYSIS")
        print("="*60)
        
        # Calculate narrative lengths in words
        def count_words(text):
            if pd.isna(text):
                return 0
            text_str = str(text).strip()
            if text_str == '' or text_str.lower() in ['nan', 'null', 'none', 'na']:
                return 0
            # Simple word count by splitting on whitespace
            return len(text_str.split())
        
        self.df['narrative_length_words'] = self.df[self.narrative_column].apply(count_words)
        
        # Filter out zero-length narratives
        valid_narratives = self.df[self.df['narrative_length_words'] > 0]
        
        length_stats = {
            'total_with_narrative': len(valid_narratives),
            'total_without_narrative': len(self.df) - len(valid_narratives),
            'percent_with_narrative': (len(valid_narratives) / len(self.df)) * 100,
            'mean_length': valid_narratives['narrative_length_words'].mean(),
            'median_length': valid_narratives['narrative_length_words'].median(),
            'std_length': valid_narratives['narrative_length_words'].std(),
            'min_length': valid_narratives['narrative_length_words'].min(),
            'max_length': valid_narratives['narrative_length_words'].max(),
            'q1': valid_narratives['narrative_length_words'].quantile(0.25),
            'q3': valid_narratives['narrative_length_words'].quantile(0.75)
        }
        
        print(f"\nNarrative Statistics (in words):")
        print(f"Complaints with narrative: {length_stats['total_with_narrative']:,} ({length_stats['percent_with_narrative']:.1f}%)")
        print(f"Complaints without narrative: {length_stats['total_without_narrative']:,}")
        print(f"Mean length: {length_stats['mean_length']:.1f} words")
        print(f"Median length: {length_stats['median_length']:.1f} words")
        print(f"Standard deviation: {length_stats['std_length']:.1f} words")
        print(f"Range: {length_stats['min_length']} to {length_stats['max_length']} words")
        print(f"25th percentile: {length_stats['q1']:.1f} words")
        print(f"75th percentile: {length_stats['q3']:.1f} words")
        
        # Create visualizations
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Histogram
        axes[0].hist(valid_narratives['narrative_length_words'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(length_stats['mean_length'], color='red', linestyle='--', label=f'Mean: {length_stats["mean_length"]:.1f}')
        axes[0].axvline(length_stats['median_length'], color='green', linestyle='--', label=f'Median: {length_stats["median_length"]:.1f}')
        axes[0].set_xlabel('Narrative Length (words)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Narrative Lengths')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(valid_narratives['narrative_length_words'], vert=False)
        axes[1].set_xlabel('Narrative Length (words)')
        axes[1].set_title('Box Plot of Narrative Lengths')
        axes[1].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_lengths = np.sort(valid_narratives['narrative_length_words'])
        cum_dist = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
        axes[2].plot(sorted_lengths, cum_dist)
        axes[2].set_xlabel('Narrative Length (words)')
        axes[2].set_ylabel('Cumulative Probability')
        axes[2].set_title('Cumulative Distribution of Narrative Lengths')
        axes[2].grid(True, alpha=0.3)
        
        # Add percentile lines
        for percentile, value in [(25, length_stats['q1']), (50, length_stats['median_length']), (75, length_stats['q3'])]:
            axes[2].axvline(value, color='red', linestyle='--', alpha=0.5)
            axes[2].text(value, 0.5, f'{percentile}%', rotation=90, va='center')
        
        plt.tight_layout()
        plt.savefig('data/processed/narrative_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyze extremes
        short_threshold = 10
        long_threshold = 500
        
        very_short = valid_narratives[valid_narratives['narrative_length_words'] <= short_threshold]
        very_long = valid_narratives[valid_narratives['narrative_length_words'] >= long_threshold]
        
        print(f"\nExtreme Length Analysis:")
        print(f"Very short narratives (≤ {short_threshold} words): {len(very_short):,} ({len(very_short)/len(valid_narratives)*100:.1f}%)")
        print(f"Very long narratives (≥ {long_threshold} words): {len(very_long):,} ({len(very_long)/len(valid_narratives)*100:.1f}%)")
        
        if len(very_short) > 0:
            print(f"\nSample of very short narratives:")
            for i, (idx, row) in enumerate(very_short.head(3).iterrows()):
                print(f"\n{i+1}. Length: {row['narrative_length_words']} words")
                narrative_preview = str(row[self.narrative_column])[:150]
                print(f"   Preview: {narrative_preview}...")
        
        length_stats['very_short_count'] = len(very_short)
        length_stats['very_long_count'] = len(very_long)
        
        return length_stats
    
    def filter_dataset(self) -> pd.DataFrame:
        """
        Filter dataset to meet project requirements
        
        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n" + "="*60)
        print("DATASET FILTERING")
        print("="*60)
        
        filtered_df = self.df.copy()
        original_count = len(filtered_df)
        
        print(f"\nStarting with {original_count:,} total complaints")
        
        # Step 1: Filter for target products (case-insensitive and flexible)
        if self.product_column:
            print(f"\n1. Filtering for target products/services:")
            print(f"   Target: {self.target_products}")
            
            # Create case-insensitive search patterns
            product_patterns = []
            for product in self.target_products:
                # Create flexible patterns
                patterns = [
                    product.lower(),
                    product.replace(' ', '').lower(),
                    product.replace(' ', '-').lower(),
                    product.replace(' ', '_').lower()
                ]
                product_patterns.extend(patterns)
            
            # Create filter mask
            product_mask = filtered_df[self.product_column].astype(str).str.lower().apply(
                lambda x: any(pattern in x for pattern in product_patterns)
            )
            
            filtered_df = filtered_df[product_mask].copy()
            print(f"   After product filtering: {len(filtered_df):,} complaints ({len(filtered_df)/original_count*100:.1f}%)")
        
        # Step 2: Remove empty/invalid narratives
        if self.narrative_column:
            print(f"\n2. Removing empty/invalid narratives:")
            
            def is_valid_narrative(text):
                if pd.isna(text):
                    return False
                text_str = str(text).strip()
                if text_str == '':
                    return False
                if text_str.lower() in ['nan', 'null', 'none', 'na', 'n/a', 'not provided', 'no description']:
                    return False
                # Check for meaningful content (at least 3 words)
                words = text_str.split()
                if len(words) < 3:
                    return False
                # Check if it's not just special characters
                if not any(char.isalnum() for char in text_str):
                    return False
                return True
            
            narrative_mask = filtered_df[self.narrative_column].apply(is_valid_narrative)
            filtered_df = filtered_df[narrative_mask].copy()
            
            print(f"   After narrative filtering: {len(filtered_df):,} complaints ({len(filtered_df)/original_count*100:.1f}%)")
        
        # Step 3: Standardize product names
        if self.product_column and len(filtered_df) > 0:
            print(f"\n3. Standardizing product/service names:")
            
            # Map variations to standard names
            product_mapping = {}
            for target in self.target_products:
                # Find rows that might belong to this target
                target_lower = target.lower()
                variations = filtered_df[
                    filtered_df[self.product_column].astype(str).str.lower().str.contains(target_lower)
                ][self.product_column].unique()
                
                for var in variations:
                    product_mapping[var] = target
            
            # Apply mapping
            if product_mapping:
                filtered_df[self.product_column] = filtered_df[self.product_column].map(product_mapping).fillna(filtered_df[self.product_column])
                print(f"   Standardized {len(product_mapping)} product/service names")
            
            # Show final distribution
            print(f"\n   Final product/service distribution:")
            final_dist = filtered_df[self.product_column].value_counts()
            for product, count in final_dist.items():
                print(f"   - {product}: {count:,} complaints ({count/len(filtered_df)*100:.1f}%)")
        
        self.filtered_df = filtered_df
        print(f"\nFiltering complete! Final dataset: {len(filtered_df):,} complaints")
        
        return filtered_df
    
    def clean_text_narratives(self) -> pd.DataFrame:
        """
        Clean text narratives to improve embedding quality
        
        Returns:
            DataFrame with cleaned narratives
        """
        if self.filtered_df is None:
            raise ValueError("Filtered data not available. Call filter_dataset() first.")
        
        if self.narrative_column is None:
            print("No narrative column to clean")
            return self.filtered_df
        
        print("\n" + "="*60)
        print("TEXT CLEANING")
        print("="*60)
        
        cleaned_df = self.filtered_df.copy()
        
        # Create new column for cleaned narratives
        cleaned_col = 'cleaned_narrative'
        
        def clean_text(text):
            """Clean individual text narrative"""
            if pd.isna(text):
                return ""
            
            text_str = str(text)
            
            # 1. Lowercase
            text_str = text_str.lower()
            
            # 2. Remove common boilerplate text (ticket/issue specific)
            boilerplate_patterns = [
                r'^dear (sir|madam|team|support|customer service)',
                r'^to whom it may concern',
                r'^hi,? ',
                r'^hello,? ',
                r'^good (morning|afternoon|evening)',
                r'^i am writing (to|because|regarding|about)',
                r'^this is (regarding|about|to report|to inform)',
                r'^i would like to (report|complain|inform|notify)',
                r'^please be (advised|informed|noted)',
                r'^kindly (note|be informed|be advised)',
                r'^regarding (my|our|the)',
                r'^reference:?.*$',
                r'^case (id|number|no):?.*$',
                r'^ticket (id|number|no):?.*$',
                r'^issue (id|number|no):?.*$',
                r'^complaint (id|number|no):?.*$',
                r'^incident (id|number|no):?.*$',
                r'^date:? \d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}',
                r'^time:? \d{1,2}:\d{2}',
                r'^account (id|number|no):?[\s\w\*]+',
                r'^customer (id|number|no):?[\s\w\*]+',
                r'^phone:?[\s\d\-\(\)]+',
                r'^email:?[\s\w\.\-@]+',
                r'^address:?[\s\w\.,\-]+',
                r'\bthank you\b.*$',
                r'\bbest regards\b.*$',
                r'\bsincerely\b.*$',
                r'\bplease help\b.*$',
                r'\blooking forward to your (response|reply)\b.*$',
                r'xxxx',  # Redacted information
                r'\*{4,}',  # Multiple asterisks
                r'\[.*?\]',  # Anything in brackets
                r'\(.*?\)',  # Anything in parentheses
            ]
            
            for pattern in boilerplate_patterns:
                text_str = re.sub(pattern, '', text_str, flags=re.IGNORECASE | re.MULTILINE)
            
            # 3. Remove URLs
            text_str = re.sub(r'https?://\S+|www\.\S+', '', text_str)
            
            # 4. Remove email addresses
            text_str = re.sub(r'\S+@\S+', '', text_str)
            
            # 5. Remove special characters but keep basic punctuation and letters
            text_str = re.sub(r'[^\w\s.,!?;:\-\'"()]', ' ', text_str)
            
            # 6. Remove extra whitespace
            text_str = re.sub(r'\s+', ' ', text_str).strip()
            
            # 7. Remove very short sentences (artifacts)
            sentences = re.split(r'[.!?]+', text_str)
            valid_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence.split()) >= 3:  # At least 3 words
                    valid_sentences.append(sentence)
            
            text_str = '. '.join(valid_sentences)
            
            return text_str
        
        print("Cleaning narratives...")
        cleaned_df[cleaned_col] = cleaned_df[self.narrative_column].apply(clean_text)
        
        # Calculate cleaning statistics
        def count_original_words(text):
            if pd.isna(text):
                return 0
            return len(str(text).split())
        
        def count_cleaned_words(text):
            if pd.isna(text) or text == "":
                return 0
            return len(text.split())
        
        original_lengths = cleaned_df[self.narrative_column].apply(count_original_words)
        cleaned_lengths = cleaned_df[cleaned_col].apply(count_cleaned_words)
        
        print(f"\nCleaning Statistics:")
        print(f"Average original length: {original_lengths.mean():.1f} words")
        print(f"Average cleaned length: {cleaned_lengths.mean():.1f} words")
        reduction_pct = ((original_lengths - cleaned_lengths) / original_lengths.replace(0, 1) * 100).mean()
        print(f"Average reduction: {reduction_pct:.1f}%")
        
        # Show before/after samples
        print(f"\nSample Cleaning Results:")
        sample_indices = cleaned_df.head(3).index
        
        for i, idx in enumerate(sample_indices):
            original = str(cleaned_df.loc[idx, self.narrative_column])
            cleaned = str(cleaned_df.loc[idx, cleaned_col])
            
            print(f"\n{'='*50}")
            print(f"SAMPLE {i+1}:")
            print(f"{'='*50}")
            print(f"\nORIGINAL (first 200 chars):")
            print(f"{original[:200]}...")
            print(f"\nCLEANED (first 200 chars):")
            print(f"{cleaned[:200]}...")
            print(f"\nOriginal length: {len(original):,} chars, {len(original.split())} words")
            print(f"Cleaned length: {len(cleaned):,} chars, {len(cleaned.split())} words")
        
        self.filtered_df = cleaned_df
        
        # Check for empty cleaned narratives
        empty_cleaned = cleaned_df[cleaned_col].apply(lambda x: x == "").sum()
        if empty_cleaned > 0:
            print(f"\nWarning: {empty_cleaned} narratives became empty after cleaning")
        
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
        print(f"Columns: {list(self.filtered_df.columns)}")
        
        # Also save as pickle for faster loading
        pickle_path = output_path.replace('.csv', '.pkl')
        self.filtered_df.to_pickle(pickle_path)
        print(f"Also saved as pickle: {pickle_path}")
        
        return output_path
    
    def generate_eda_report(self, output_path: str = 'data/processed/eda_report.txt'):
        """
        Generate a comprehensive EDA report
        
        Args:
            output_path: Path to save the report
        """
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("EXPLORATORY DATA ANALYSIS REPORT - COMPLAINT TICKET DATA")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {pd.Timestamp.now()}")
        report_lines.append("\n")
        
        # Dataset overview
        report_lines.append("1. DATASET OVERVIEW")
        report_lines.append("-" * 40)
        report_lines.append(f"Source file: {self.data_path}")
        report_lines.append(f"Total records: {len(self.df):,}")
        report_lines.append(f"Total columns: {len(self.df.columns)}")
        report_lines.append(f"Missing values: {self.df.isnull().sum().sum():,}")
        report_lines.append(f"Missing percentage: {(self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1]) * 100):.2f}%")
        
        # Key columns
        report_lines.append("\n2. KEY COLUMNS IDENTIFIED")
        report_lines.append("-" * 40)
        if self.product_column:
            report_lines.append(f"Product/Service column: {self.product_column}")
            report_lines.append(f"  Unique values: {self.df[self.product_column].nunique()}")
        if self.narrative_column:
            report_lines.append(f"Narrative column: {self.narrative_column}")
        
        # Filtered dataset info
        if self.filtered_df is not None:
            report_lines.append("\n3. FILTERED DATASET")
            report_lines.append("-" * 40)
            report_lines.append(f"Total filtered complaints: {len(self.filtered_df):,}")
            
            if self.product_column and self.product_column in self.filtered_df.columns:
                report_lines.append("\nProduct/Service distribution in filtered data:")
                product_dist = self.filtered_df[self.product_column].value_counts()
                for product, count in product_dist.items():
                    percentage = (count / len(self.filtered_df)) * 100
                    report_lines.append(f"  - {product}: {count:,} complaints ({percentage:.1f}%)")
        
        # Narrative analysis
        if 'narrative_length_words' in self.df.columns:
            report_lines.append("\n4. NARRATIVE ANALYSIS")
            report_lines.append("-" * 40)
            valid_narratives = self.df[self.df['narrative_length_words'] > 0]
            report_lines.append(f"Complaints with narrative: {len(valid_narratives):,} ({len(valid_narratives)/len(self.df)*100:.1f}%)")
            report_lines.append(f"Complaints without narrative: {len(self.df) - len(valid_narratives):,}")
            report_lines.append(f"Average narrative length: {valid_narratives['narrative_length_words'].mean():.1f} words")
            report_lines.append(f"Median narrative length: {valid_narratives['narrative_length_words'].median():.1f} words")
        
        # Key findings
        report_lines.append("\n5. KEY FINDINGS")
        report_lines.append("-" * 40)
        report_lines.append("1. Dataset contains complaint/ticket data with textual narratives.")
        report_lines.append("2. Narrative lengths vary significantly, requiring appropriate chunking strategy.")
        report_lines.append("3. Common boilerplate text found in narratives needs removal for better embeddings.")
        report_lines.append("4. Filtering ensures focus on relevant product/service categories.")
        report_lines.append("5. Text cleaning improves embedding quality by removing noise.")
        
        # Recommendations for RAG pipeline
        report_lines.append("\n6. RECOMMENDATIONS FOR RAG PIPELINE")
        report_lines.append("-" * 40)
        report_lines.append("1. Use chunk_size=500 and chunk_overlap=50 for balanced context preservation.")
        report_lines.append("2. Implement metadata tracking (product, issue type, dates) for better retrieval.")
        report_lines.append("3. Consider stratified sampling for balanced representation across categories.")
        report_lines.append("4. Use sentence-transformers/all-MiniLM-L6-v2 for efficient embeddings.")
        report_lines.append("5. Store vectors with FAISS for fast similarity search.")
        
        # Data quality issues
        report_lines.append("\n7. DATA QUALITY ISSUES IDENTIFIED")
        report_lines.append("-" * 40)
        if self.narrative_column:
            empty_narratives = self.df[self.narrative_column].isna() | (self.df[self.narrative_column].astype(str).str.strip() == '')
            report_lines.append(f"- Empty narratives: {empty_narratives.sum():,} ({empty_narratives.sum()/len(self.df)*100:.1f}%)")
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\nEDA report saved to: {output_path}")
        
        # Also save as markdown
        md_path = output_path.replace('.txt', '.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Markdown report saved to: {md_path}")
        
        return '\n'.join(report_lines)

def main():
    """Main execution function"""
    print("=" * 80)
    print("COMPLAINT DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Initialize with your specific path
    data_path = r'C:\Users\admin\Rag-complaint-chatbot\data\raw\complaints.csv'
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"\nERROR: File not found at: {data_path}")
        print("Please ensure the complaints.csv file exists at the specified path.")
        print("\nPossible solutions:")
        print("1. Check if the file path is correct")
        print("2. Make sure the file exists in the data/raw/ directory")
        print("3. Download the data if not already available")
        return
    
    print(f"\nUsing data from: {data_path}")
    
    # Initialize preprocessor
    preprocessor = ComplaintDataPreprocessor(data_path=data_path)
    
    # Load data
    print("\n[1/6] Loading data...")
    df = preprocessor.load_data()
    
    if df is None or len(df) == 0:
        print("Failed to load data or data is empty. Exiting.")
        return
    
    # Perform EDA
    print("\n[2/6] Performing exploratory data analysis...")
    eda_stats = preprocessor.perform_initial_eda()
    
    # Analyze product distribution
    print("\n[3/6] Analyzing product/service distribution...")
    product_dist = preprocessor.analyze_product_distribution()
    
    # Analyze narrative lengths
    print("\n[4/6] Analyzing narrative lengths...")
    length_stats = preprocessor.analyze_narrative_lengths()
    
    # Filter dataset
    print("\n[5/6] Filtering dataset...")
    filtered_df = preprocessor.filter_dataset()
    
    if filtered_df is not None and len(filtered_df) > 0:
        # Clean narratives
        print("\n[6/6] Cleaning text narratives...")
        cleaned_df = preprocessor.clean_text_narratives()
        
        # Save filtered data
        output_path = preprocessor.save_filtered_data()
        
        # Generate EDA report
        report = preprocessor.generate_eda_report()
        
        print("\n" + "="*80)
        print("PREPROCESSING COMPLETE!")
        print("="*80)
        print(f"\nSummary:")
        print(f"  • Original dataset: {len(df):,} complaints")
        print(f"  • Filtered dataset: {len(cleaned_df):,} complaints")
        print(f"  • Saved to: {output_path}")
        print(f"  • EDA report generated: data/processed/eda_report.txt")
        print("\nNext steps: Run Task 2 for text chunking and embedding.")
    else:
        print("\nERROR: No data after filtering. Please check your data and filtering criteria.")

if __name__ == "__main__":
    main()