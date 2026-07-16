import streamlit as st
import os
import torch
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'geo-clip')))
from geoclip.model.GeoCLIP import GeoCLIP

st.set_page_config(page_title="GeoCLIP AI Locator", page_icon="🌍", layout="centered")

@st.cache_resource
def load_models_and_index():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoCLIP().to(device)
    model.eval()
    
    global_index = faiss.read_index("data/global_index.faiss")
    global_cities = pd.read_csv("data/global_cities.csv")
    
    iraq_index = None
    iraq_cities = None
    if os.path.exists("data/iraq_index.faiss"):
        iraq_index = faiss.read_index("data/iraq_index.faiss")
        iraq_cities = pd.read_csv("data/iraq_locations.csv")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return device, model, transform, global_index, global_cities, iraq_index, iraq_cities

def get_predictions(image, use_iraq=False, top_k=3):
    device, model, transform, global_index, global_cities, iraq_index, iraq_cities = load_models_and_index()
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        img_features = model.image_encoder(img_tensor)
        img_features = img_features.cpu().numpy()
        
    if use_iraq and iraq_index is not None:
        index = iraq_index
        cities = iraq_cities
    else:
        index = global_index
        cities = global_cities
        
    distances, indices = index.search(img_features, top_k)
    
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        dist = distances[0][i]
        row = cities.iloc[idx]
        
        # Calculate a mock confidence score from L2 distance
        conf = max(0, 100 - (dist * 10))
        
        # Format name properly whether it's a City or Grid Point
        name = row.get('City', 'Unknown')
        if pd.isna(name):
            name = "Grid Location"
            
        results.append({
            "name": name,
            "country": row.get('CountryCode', 'Unknown'),
            "lat": row['LAT'],
            "lon": row['LON'],
            "distance": dist,
            "confidence": f"{conf:.1f}%"
        })
        
    return results

def main():
    st.title("🌍 AI Geo-Locator")
    st.markdown("Upload a photo and the AI will predict its geographic location based on visual features.")
    
    with st.spinner("Initializing AI Core..."):
        device, _, _, _, _, iraq_idx, _ = load_models_and_index()
        
    st.sidebar.markdown("### Model Settings")
    st.sidebar.markdown(f"**Hardware:** `{device}`")
    
    use_iraq = False
    if iraq_idx is not None:
        use_iraq = st.sidebar.checkbox("🇮🇶 Use Iraq-Only Database", value=True, help="Force the AI to only search within Iraqi borders.")
        
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("📍 Predict Location", type="primary"):
            with st.spinner("Analyzing visual features and searching database..."):
                results = get_predictions(image, use_iraq=use_iraq)
                
            st.success("Analysis Complete!")
            
            st.subheader("Top Predictions")
            for i, res in enumerate(results):
                country_label = res['country'] if not pd.isna(res['country']) else 'GRID'
                with st.container():
                    st.markdown(f"""
                    <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #333;">
                        <h4 style="margin:0; color: #4DA8DA;">#{i+1}: {res['name']}, {country_label}</h4>
                        <p style="margin: 5px 0; font-size: 14px; color: #ccc;">
                            <b>Latitude:</b> {res['lat']:.4f} <br>
                            <b>Longitude:</b> {res['lon']:.4f} <br>
                            <b>Confidence Score:</b> {res['confidence']}
                        </p>
                        <a href="https://www.google.com/maps?q={res['lat']},{res['lon']}" target="_blank" style="text-decoration: none;">
                            <button style="background-color: #4CAF50; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer; margin-top: 10px;">
                                🗺️ View on Google Maps
                            </button>
                        </a>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
