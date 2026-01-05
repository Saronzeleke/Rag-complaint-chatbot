# tests/test_preprocessing.py
import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import ComplaintDataPreprocessor
from src.embedding_pipeline import EmbeddingPipeline

def test_preprocessor_initialization():
    """Test that preprocessor initializes correctly"""
    preprocessor = ComplaintDataPreprocessor()
    assert preprocessor.target_products == [
        'Credit card', 
        'Personal loan', 
        'Savings account', 
        'Money transfers'
    ]
    assert preprocessor.df is None
    assert preprocessor.filtered_df is None

def test_embedding_pipeline_initialization():
    """Test that embedding pipeline initializes correctly"""
    pipeline = EmbeddingPipeline()
    assert pipeline.df is None
    assert pipeline.sampled_df is None
    assert pipeline.chunks == []
    assert pipeline.vector_store is None

def test_sample_dataframe_creation():
    """Test creation of sample DataFrame for testing"""
    data = {
        'Product': ['Credit card', 'Personal loan', 'Credit card', 'Savings account'],
        'Consumer complaint narrative': [
            'This is a test complaint about credit card.',
            'I have issues with my personal loan.',
            'Another credit card complaint.',
            'Savings account problem here.'
        ]
    }
    df = pd.DataFrame(data)
    assert len(df) == 4
    assert 'Product' in df.columns
    assert 'Consumer complaint narrative' in df.columns

if __name__ == '__main__':
    pytest.main([__file__, '-v'])