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
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    
    return device, model, transform, global_index, global_cities, iraq_index, iraq_cities

@st.cache_resource
def load_ocr_reader():
    # Load EasyOCR for Arabic and English
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
    
    # We use A4 variant (TTA + Micro-Grid + OCR) for the best results in the app
    results = engine.predict(image, variant="A4", top_k=top_k)
    
    formatted_results = []
    for res in results:
        # If it's a coarse result it has no confidence_prob, if micro-grid it has.
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
        if not exif:
            return None
            
        geotagging = {}
        for (idx, tag) in ExifTags.TAGS.items():
            if tag == 'GPSInfo':
                if idx not in exif:
                    return None
                for (key, val) in ExifTags.GPSTAGS.items():
                    if key in exif[idx]:
                        geotagging[val] = exif[idx][key]
                break
                
        if not geotagging or 'GPSLatitude' not in geotagging or 'GPSLongitude' not in geotagging:
            return None

        def get_decimal_from_dms(dms, ref):
            degrees = float(dms[0])
            minutes = float(dms[1])
            seconds = float(dms[2])
            
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            if ref in ['S', 'W']:
                decimal = -decimal
            return decimal

        lat = get_decimal_from_dms(geotagging['GPSLatitude'], geotagging['GPSLatitudeRef'])
        lon = get_decimal_from_dms(geotagging['GPSLongitude'], geotagging['GPSLongitudeRef'])
        return lat, lon
    except Exception as e:
        return None

def main():
    import sys
    use_forensic = "--use-exif-oracle" in sys.argv
    
    st.title("🌍 AI Geo-Locator")
    st.markdown("Upload a photo and the AI will predict its geographic location based on visual features.")
    
    with st.spinner("Initializing AI Core & OCR..."):
        device, _, _, _, _, iraq_idx, _ = load_models_and_index()
        ocr_reader = load_ocr_reader()
        
    st.sidebar.markdown("### Model Settings")
    st.sidebar.markdown(f"**Hardware:** `{device}`")
    
    use_iraq = False
    if iraq_idx is not None:
        use_iraq = st.sidebar.checkbox("🇮🇶 Use Iraq-Only Database", value=True, help="Force the AI to only search within Iraqi borders.")
        
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file)
        
        exif_loc = None
        if use_forensic:
            exif_loc = get_exif_location(raw_image)
            
        image = raw_image.convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("📍 Predict Location", type="primary"):
            # Check for EXIF data first (Only if --use-exif-oracle flag is enabled)
            
            if exif_loc:
                st.success("✅ Exact Location Found from Image Metadata (EXIF)!")
                st.markdown(f"""
                <div style="background-color: #2e7d32; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #4CAF50;">
                    <h4 style="margin:0; color: #fff;">🎯 100% Accurate (GPS Metadata)</h4>
                    <p style="margin: 5px 0; font-size: 14px; color: #eee;">
                        <b>Latitude:</b> {exif_loc[0]:.6f} <br>
                        <b>Longitude:</b> {exif_loc[1]:.6f} <br>
                        <b>Confidence Score:</b> 100%
                    </p>
                    <a href="https://www.google.com/maps?q={exif_loc[0]},{exif_loc[1]}" target="_blank" style="text-decoration: none;">
                        <button style="background-color: #fff; color: #2e7d32; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer; margin-top: 10px; font-weight: bold;">
                            🗺️ View Exact Location on Google Maps
                        </button>
                    </a>
                </div>
                """, unsafe_allow_html=True)
                st.info("The predictions below are the AI's guesses based on visual features, shown for comparison.")
            
            # --- OCR Extraction ---
            with st.spinner("Extracting text from image (OCR)..."):
                img_np = np.array(image)
                ocr_results = ocr_reader.readtext(img_np)
                detected_text = [res[1] for res in ocr_results if res[2] > 0.35]
                
            if detected_text:
                st.markdown(f"""
                <div style="background-color: #2b2b2b; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #4DA8DA;">
                    <h4 style="margin:0; color: #4DA8DA;">📝 Text Detected (OCR Auxiliary Cue)</h4>
                    <p style="margin: 5px 0; font-size: 14px; color: #eee;">
                        The AI read the following text from the image which might help identify the location:<br>
                        <b>{', '.join(detected_text)}</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            with st.spinner("Analyzing visual features and searching database..."):
                results = get_predictions(image, use_iraq=use_iraq)
                
            if not exif_loc:
                st.success("Analysis Complete!")
                
            st.subheader("Top AI Predictions (Visual Features)")
            for i, res in enumerate(results):
                country_label = res['country'] if not pd.isna(res['country']) else 'GRID'
                with st.container():
                    st.markdown(f"""
                    <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #333;">
                        <h4 style="margin:0; color: #4DA8DA;">#{i+1}: {res['name']}, {country_label}</h4>
                        <p style="margin: 5px 0; font-size: 14px; color: #ccc;">
                            <b>Latitude:</b> {res['lat']:.4f} <br>
                            <b>Longitude:</b> {res['lon']:.4f} <br>
                            {f"<b>Confidence:</b> {res['confidence']}" if res['confidence'] != 'N/A' else ""}
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
