import streamlit as st
import os
import torch
import faiss
import numpy as np
import pandas as pd
from PIL import Image, ExifTags
from torchvision import transforms
import easyocr
import sys

# Windows Encoding Fix for Emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Configure page layout and icon
st.set_page_config(page_title="AI Geo-Locator (Unified System)", page_icon="🌍", layout="wide")

# ==========================================
# PREMIUM UI STYLING (Vanilla CSS Injection)
# ==========================================
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Base styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Global Background */
    .stApp {
        background: radial-gradient(circle at 15% 50%, #0d1117, #010409 60%, #000000 100%);
        color: #e6edf3;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        background: -webkit-linear-gradient(45deg, #4DA8DA, #00ffcc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Glassmorphism Containers */
    .glass-panel {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-panel:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0, 255, 204, 0.1);
        border: 1px solid rgba(0, 255, 204, 0.2);
    }
    
    /* Custom Button */
    .stButton > button {
        background: linear-gradient(135deg, #00c6ff, #0072ff);
        color: white !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3) !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(0, 114, 255, 0.5) !important;
    }

    /* Subheaders inside glass panels */
    .glass-panel h4 {
        margin-top: 0;
        margin-bottom: 12px;
        font-size: 1.2rem;
    }
    
    /* Data labels */
    .data-label {
        color: #8b949e;
        font-size: 0.9rem;
        margin-bottom: 2px;
    }
    
    .data-value {
        color: #ffffff;
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 12px;
    }
    
    /* Map Button */
    .map-btn {
        display: inline-block;
        background: rgba(46, 160, 67, 0.1);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.4);
        padding: 8px 16px;
        border-radius: 6px;
        text-decoration: none;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        text-align: center;
        margin-top: 10px;
    }
    
    .map-btn:hover {
        background: rgba(46, 160, 67, 0.2);
        box-shadow: 0 0 10px rgba(46, 160, 67, 0.2);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(13, 17, 23, 0.7) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'geo-clip')))
from geoclip.model.GeoCLIP import GeoCLIP

@st.cache_resource
def load_models_and_index():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoCLIP().to(device)
    model.eval()
    
    global_cities = pd.read_csv("data/global_cities.csv")
    if os.path.exists("data/global_index.faiss"):
        global_index = faiss.read_index("data/global_index.faiss")
    else:
        coords = np.stack([global_cities['LAT'].values, global_cities['LON'].values], axis=1)
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        batch_size = 1024
        features_list = []
        with torch.no_grad():
            for i in range(0, len(coords_tensor), batch_size):
                batch = coords_tensor[i:i+batch_size].to(device)
                feats = model.location_encoder(batch)
                feats = torch.nn.functional.normalize(feats, dim=-1)
                features_list.append(feats.cpu().numpy())
        all_features = np.vstack(features_list)
        global_index = faiss.IndexFlatIP(512)
        global_index.add(all_features)
    
    iraq_index = None
    iraq_cities = None
    if os.path.exists("data/iraq_locations.csv"):
        iraq_cities = pd.read_csv("data/iraq_locations.csv")
        if os.path.exists("data/iraq_index.faiss"):
            iraq_index = faiss.read_index("data/iraq_index.faiss")
        else:
            batch_size = 1024
            coords_iq = np.stack([iraq_cities['LAT'].values, iraq_cities['LON'].values], axis=1)
            coords_iq_tensor = torch.tensor(coords_iq, dtype=torch.float32)
            features_list_iq = []
            with torch.no_grad():
                for i in range(0, len(coords_iq_tensor), batch_size):
                    batch = coords_iq_tensor[i:i+batch_size].to(device)
                    feats = model.location_encoder(batch)
                    feats = torch.nn.functional.normalize(feats, dim=-1)
                    features_list_iq.append(feats.cpu().numpy())
            all_features_iq = np.vstack(features_list_iq)
            iraq_index = faiss.IndexFlatIP(512)
            iraq_index.add(all_features_iq)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    
    return device, model, transform, global_index, global_cities, iraq_index, iraq_cities

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['ar', 'en'], gpu=torch.cuda.is_available())

def get_predictions(image, use_iraq=False, top_k=3):
    device, model, _, global_index, global_cities, iraq_index, iraq_cities = load_models_and_index()
    ocr_reader = load_ocr_reader()
    
    from src.inference import InferenceEngine
    
    if use_iraq and iraq_index is not None:
        index = iraq_index
        if hasattr(index, 'nprobe'):
            index.nprobe = min(16, getattr(index, 'nlist', 16))
        cities = iraq_cities
    else:
        index = global_index
        if hasattr(index, 'nprobe'):
            index.nprobe = min(64, getattr(index, 'nlist', 64))
        cities = global_cities
        
    engine = InferenceEngine(model, device, index, cities, ocr_reader)
    
    # Unified Source of Truth: We use A4 variant for maximum exploratory capability in the UI
    # Using temperature=0.35 gives a realistic, moderate confidence spread (e.g., 60-85% for strong matches)
    results = engine.predict(image, variant="A4", top_k=top_k, temperature=0.35)
    
    formatted_results = []
    for res in results:
        conf_str = f"{res['confidence_prob']:.1f}%" if res['confidence_prob'] > 0 else "N/A"
        formatted_results.append({
            "name": res["name"],
            "country": res["country"],
            "lat": res["lat"],
            "lon": res["lon"],
            "distance": res["score"],
            "confidence": conf_str,
            "ocr_text": res["ocr_text"]
        })
    return formatted_results

def get_exif_location(image):
    try:
        exif = image._getexif()
        if not exif: return None
            
        geotagging = {}
        for (idx, tag) in ExifTags.TAGS.items():
            if tag == 'GPSInfo':
                if idx not in exif: return None
                for (key, val) in ExifTags.GPSTAGS.items():
                    if key in exif[idx]:
                        geotagging[val] = exif[idx][key]
                break
                
        if not geotagging or 'GPSLatitude' not in geotagging or 'GPSLongitude' not in geotagging:
            return None

        def get_decimal_from_dms(dms, ref):
            degrees = float(dms[0]); minutes = float(dms[1]); seconds = float(dms[2])
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            if ref in ['S', 'W']: decimal = -decimal
            return decimal

        lat = get_decimal_from_dms(geotagging['GPSLatitude'], geotagging['GPSLatitudeRef'])
        lon = get_decimal_from_dms(geotagging['GPSLongitude'], geotagging['GPSLongitudeRef'])
        return lat, lon
    except:
        return None

def main():
    use_forensic = "--use-exif-oracle" in sys.argv
    
    st.markdown("<h1>🌍 AI Geo-Locator (Unified System)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8b949e; font-size: 1.1rem; margin-bottom: 2rem;'>Upload a photo and the AI will predict its geographic location based on visual features using the Unified GeoCLIP Pipeline.</p>", unsafe_allow_html=True)
    
    with st.spinner("Initializing AI Core & OCR..."):
        device, _, _, _, _, iraq_idx, _ = load_models_and_index()
        ocr_reader = load_ocr_reader()
        
    st.sidebar.markdown("### ⚙️ Engine Settings")
    st.sidebar.markdown(f"**Hardware Acceleration:** `<span style='color: #00ffcc;'>{device}</span>`", unsafe_allow_html=True)
    st.sidebar.markdown(f"**Architecture Mode:** `Unified A4 (TTA+OCR)`")
    
    use_iraq = False
    if iraq_idx is not None:
        st.sidebar.markdown("---")
        use_iraq = st.sidebar.checkbox("🇮🇶 Use Iraq-Only Database", value=False, help="Force the AI to only search within Iraqi borders.")
        
    # Two-column layout
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.markdown("<div class='glass-panel'><h4>1. Upload Image</h4></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            raw_image = Image.open(uploaded_file)
            exif_loc = None
            if use_forensic:
                exif_loc = get_exif_location(raw_image)
                
            image = raw_image.convert("RGB")
            st.image(image, use_container_width=True)
            
            predict_btn = st.button("📍 Initiate Inference")
            
    with col2:
        st.markdown("<div class='glass-panel'><h4>2. Analysis & Results</h4></div>", unsafe_allow_html=True)
        
        if uploaded_file is not None and predict_btn:
            if exif_loc:
                html_exif = f"""
<div class="glass-panel" style="border-color: rgba(46, 160, 67, 0.4); box-shadow: 0 0 20px rgba(46,160,67,0.1);">
    <h4 style="color: #3fb950; display: flex; align-items: center; gap: 8px;">
        🎯 100% Accurate (GPS Metadata)
    </h4>
    <div class="data-label">Latitude</div>
    <div class="data-value">{exif_loc[0]:.6f}</div>
    <div class="data-label">Longitude</div>
    <div class="data-value">{exif_loc[1]:.6f}</div>
    <a href="https://www.google.com/maps?q={exif_loc[0]},{exif_loc[1]}" target="_blank" class="map-btn">
        🗺️ View Exact Location
    </a>
</div>
"""
                st.markdown(html_exif, unsafe_allow_html=True)
            
            with st.spinner("Extracting text cues (OCR)..."):
                img_np = np.array(image)
                ocr_results = ocr_reader.readtext(img_np)
                detected_text = [res[1] for res in ocr_results if res[2] > 0.35]
                
            if detected_text:
                html_ocr = f"""
<div class="glass-panel" style="border-left: 4px solid #4DA8DA;">
    <h4 style="color: #4DA8DA;">📝 Auxiliary Cues (OCR)</h4>
    <p style="color: #c9d1d9; font-size: 0.95rem;">
        Detected Text: <b>{', '.join(detected_text)}</b>
    </p>
</div>
"""
                st.markdown(html_ocr, unsafe_allow_html=True)
                
            with st.spinner("Executing Unified Visual Feature Search..."):
                results = get_predictions(image, use_iraq=use_iraq)
                
            st.markdown("<h3 style='margin-top: 1.5rem; margin-bottom: 1rem;'>Top Predictions</h3>", unsafe_allow_html=True)
            
            for i, res in enumerate(results):
                country_label = res['country'] if not pd.isna(res['country']) else 'GRID'
                
                # Visual hierarchy for top result vs others
                border_color = "rgba(0, 255, 204, 0.3)" if i == 0 else "rgba(255, 255, 255, 0.08)"
                glow = "box-shadow: 0 0 15px rgba(0, 255, 204, 0.1);" if i == 0 else ""
                
                conf_html = f"""<div class="data-label">Model Confidence</div><div class="data-value">{res['confidence']}</div>""" if res['confidence'] != 'N/A' else ""
                
                html_str = f"""
<div class="glass-panel" style="border: 1px solid {border_color}; {glow}">
    <h4 style="color: {'#00ffcc' if i==0 else '#e6edf3'};">#{i+1}: {res['name']}, {country_label}</h4>
    <div style="display: grid; grid-template-columns: 1fr 1fr; margin-top: 15px;">
        <div>
            <div class="data-label">Latitude</div>
            <div class="data-value">{res['lat']:.4f}</div>
        </div>
        <div>
            <div class="data-label">Longitude</div>
            <div class="data-value">{res['lon']:.4f}</div>
        </div>
    </div>
    {conf_html}
    <a href="https://www.google.com/maps?q={res['lat']},{res['lon']}" target="_blank" class="map-btn" style="margin-top: 15px;">
        🗺️ View on Map
    </a>
</div>
"""
                st.markdown(html_str, unsafe_allow_html=True)

        elif not uploaded_file:
            st.info("Upload an image on the left to begin analysis.")

if __name__ == "__main__":
    main()
