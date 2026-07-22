import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import MagicMock
from src.inference import InferenceEngine

# Mock the model, device, index and cities
@pytest.fixture
def mock_engine():
    device = "cpu"
    
    # Mock model
    class MockModel:
        def __init__(self):
            self.logit_scale = torch.nn.Parameter(torch.tensor(4.6052))
            
        def predict(self, *args, **kwargs):
            return torch.rand((1, 512))
            
        def location_encoder(self, *args, **kwargs):
            return torch.rand((1, 512))
            
    model = MockModel()
    
    # Mock FAISS index
    index = MagicMock()
    index.search.return_value = (np.array([[10.0, 5.0, 2.0]]), np.array([[0, 1, 2]]))
    
    # Mock cities dataframe
    import pandas as pd
    cities_df = pd.DataFrame({
        "LAT": [35.0, 36.0, 37.0],
        "LON": [139.0, 140.0, 141.0],
        "CITY": ["Tokyo", "Osaka", "Kyoto"],
        "COUNTRY": ["Japan", "Japan", "Japan"]
    })
    
    return InferenceEngine(model=model, device=device, index=index, cities_df=cities_df)

def test_inference_baseline(mock_engine):
    # Load a mock image from tests/fixtures
    image = Image.open('tests/fixtures/mock/mock_ref_1.jpg').convert('RGB')
    
    # Run prediction for A1
    results = mock_engine.predict(image, variant="A1", top_k=3)
    
    assert len(results) == 3
    assert "lat" in results[0]
    assert "lon" in results[0]
    assert "score" in results[0]

def test_inference_microgrid(mock_engine):
    # Load a mock image from tests/fixtures
    image = Image.open('tests/fixtures/mock/mock_ref_1.jpg').convert('RGB')
    
    # Run prediction for A3 (Micro-Grid)
    # The mock model's location_encoder will be called
    results = mock_engine.predict(image, variant="A3", top_k=1)
    
    assert len(results) == 1
    assert "confidence_prob" in results[0]
