"""Build a Middle East gallery using GeoCLIP's location encoder.
Instead of downloading images, we use GeoCLIP's own location encoder to embed
GPS coordinates directly, creating a dense Middle East gallery."""
import os, sys, time, json, warnings
os.environ['OMP_NUM_THREADS'] = '8'; os.environ['MKL_NUM_THREADS'] = '8'
warnings.filterwarnings('ignore')
import numpy as np, torch
torch.set_num_threads(8)
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / 'data/raw/me_landmarks'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Famous landmarks in Iraq and surrounding countries with coordinates
# Curated from Wikipedia/Wikidata
LANDMARKS = {
    "Iraq": [
        ("Imam Husayn Shrine, Karbala", 32.6164, 44.0323),
        ("Al-Abbas Shrine, Karbala", 32.6159, 44.0336),
        ("Imam Ali Shrine, Najaf", 32.0003, 44.3145),
        ("Kadhimiya Mosque, Baghdad", 33.3534, 44.3425),
        ("Abu Hanifa Mosque, Baghdad", 33.3300, 44.3700),
        ("Al-Kadhimiya Mosque", 33.3534, 44.3425),
        ("Great Mosque of Samarra", 34.1983, 43.8744),
        ("Malwiya Minaret, Samarra", 34.1983, 43.8744),
        ("Basra Grand Mosque", 30.5086, 47.8144),
        ("Erbil Citadel", 36.1900, 44.0110),
        ("Al-Ukhaidir Fortress", 32.4300, 43.3000),
        ("Ziggurat of Ur", 30.9628, 46.1031),
        ("Ctesiphon Arch", 33.0921, 44.5816),
        ("Lion of Babylon", 33.0958, 44.3953),
        ("Hatra Ruins", 35.5878, 42.7272),
        ("Ashur (Qal'at Sherqat)", 35.4561, 43.2631),
        ("Kirkuk Citadel", 35.4681, 44.3922),
        ("Sulaymaniyah Mosque", 35.5564, 45.4352),
        ("Duhok Mosque", 36.8669, 42.9511),
        ("Nasiriyah Mosque", 31.0430, 46.2575),
        ("Amarah City Center", 31.8356, 47.1457),
        ("Diwaniyah Mosque", 31.9929, 44.9240),
        ("Hilla Historical Mosque", 32.4843, 44.4307),
        ("Ramadi Mosque", 33.4267, 43.2994),
        ("Fallujah Mosque", 33.3511, 43.7847),
        ("Mosul Great Mosque", 36.3400, 43.1300),
        ("Tigris River Baghdad", 33.3100, 44.3700),
        ("Tigris River Mosul", 36.3400, 43.1300),
        ("Euphrates River Basra", 30.5000, 47.8000),
        ("Green Zone Baghdad", 33.3100, 44.3800),
    ],
    "Iran": [
        ("Imam Reza Shrine, Mashhad", 36.2872, 59.6104),
        ("Nasir al-Mulk Mosque, Shiraz", 29.6050, 52.5539),
        ("Vakil Mosque, Shiraz", 29.6104, 52.5490),
        ("Shah Mosque, Isfahan", 32.6546, 51.6680),
        ("Ali Qapu, Isfahan", 32.6580, 51.6770),
        ("Sheikh Lotfollah Mosque", 32.6559, 51.6756),
        ("Naqsh-e Jahan Square", 32.6555, 51.6814),
        ("Persepolis", 29.9350, 52.8914),
        ("Golestan Palace, Tehran", 35.6793, 51.4044),
        ("Azadi Tower, Tehran", 35.6947, 51.3809),
        ("Tabriz Bazaar", 38.0745, 46.2922),
        ("Qom Seminary", 34.6400, 50.8800),
        ("Saveh Mosque", 35.0200, 50.3500),
        ("Qazvin Mosque", 36.2600, 50.0000),
        ("Arak Mosque", 34.0800, 49.7000),
        ("Khorramabad Mosque", 33.4872, 48.3558),
        ("Hamadan Mosque", 34.8000, 48.5200),
        ("Kerman Mosque", 30.2833, 57.0833),
        ("Yazd Mosque", 31.8972, 54.3569),
        ("Isfahan Cathedral", 32.6219, 51.6739),
    ],
    "Syria": [
        ("Umayyad Mosque, Damascus", 33.5103, 36.3069),
        ("Citadel of Damascus", 33.5125, 36.2919),
        ("Palmyra Ruins", 34.5514, 38.2672),
        ("Krak des Chevaliers", 34.7578, 36.2933),
        ("Aleppo Citadel", 36.1994, 37.1522),
        ("Great Mosque of Aleppo", 36.1983, 37.1528),
    ],
    "Jordan": [
        ("Petra Treasury", 30.3285, 35.4444),
        ("Amman Citadel", 31.9522, 35.9340),
        ("Roman Theater Amman", 31.9478, 35.9386),
        ("Wadi Rum", 29.5321, 35.4179),
        ("Jerash Ruins", 32.2789, 35.8936),
    ],
    "Lebanon": [
        ("Baalbek Ruins", 34.0047, 36.2110),
        ("Byblos Castle", 34.1211, 35.6486),
        ("Beirut Downtown", 33.8938, 35.5018),
    ],
    "Turkey": [
        ("Hagia Sophia, Istanbul", 41.0086, 28.9802),
        ("Blue Mosque, Istanbul", 41.0054, 28.9768),
        ("Cappadocia", 38.6431, 34.8289),
        ("Ephesus Ruins", 37.9411, 27.3416),
        ("Gobekli Tepe", 37.2231, 38.9225),
        ("Diyarbakir Walls", 37.9147, 40.2306),
        ("Mardin Old City", 37.3167, 40.7333),
        ("Sanliurfa Mosque", 37.1592, 38.7969),
    ],
    "Saudi Arabia": [
        ("Masjid al-Haram, Mecca", 21.4225, 39.8262),
        ("Prophet's Mosque, Medina", 24.4672, 39.6024),
        ("Kingdom Centre, Riyadh", 24.7116, 46.6753),
        ("Al-Masjid an-Nabawi", 24.4672, 39.6024),
    ],
    "Egypt": [
        ("Great Pyramid of Giza", 29.9792, 31.1342),
        ("Sphinx, Giza", 29.9753, 31.1376),
        ("Luxor Temple", 25.6999, 32.6390),
        ("Karnak Temple", 25.7188, 32.6573),
        ("Abu Simbel", 22.3360, 31.6256),
        ("Cairo Citadel", 30.0286, 31.2597),
        ("Al-Azhar Mosque", 30.0473, 31.2555),
    ],
    "Palestine": [
        ("Dome of the Rock, Jerusalem", 31.7784, 35.2355),
        ("Al-Aqsa Mosque", 31.7767, 35.2345),
        ("Church of Holy Sepulchre", 31.7781, 35.2298),
    ],
    "Kuwait": [
        ("Kuwait Towers", 29.3956, 47.9789),
        ("Grand Mosque Kuwait", 29.3750, 47.9778),
    ],
    "UAE": [
        ("Burj Khalifa, Dubai", 25.1972, 55.2744),
        ("Sheikh Zayed Mosque, Abu Dhabi", 24.4128, 54.4751),
    ],
}

print("Loading GeoCLIP location encoder...", flush=True)
from geoclip import GeoCLIP
gc_model = GeoCLIP()
gc_model.eval()

# We need the location encoder from GeoCLIP
# GeoCLIP has: model.location_encoder(gps) -> features
# and model.image_encoder(images) -> features

# Build the location embeddings
print("Building location embeddings for Middle East landmarks...", flush=True)

all_gps = []
all_names = []
all_countries = []

for country, landmarks in LANDMARKS.items():
    for name, lat, lon in landmarks:
        all_gps.append([lat, lon])
        all_names.append(name)
        all_countries.append(country)

gps_tensor = torch.tensor(all_gps, dtype=torch.float32)
print(f"Total landmarks: {len(all_gps)}", flush=True)

# Get location embeddings from GeoCLIP
with torch.no_grad():
    loc_feats = gc_model.location_encoder(gps_tensor)
    loc_feats = loc_feats / loc_feats.norm(dim=-1, keepdim=True)
    loc_feats = loc_feats.cpu().numpy().astype(np.float32)

gps_arr = np.array(all_gps, dtype=np.float32)

# Save
np.save(str(OUT_DIR / 'me_features.npy'), loc_feats)
np.save(str(OUT_DIR / 'me_gps.npy'), gps_arr)
with open(str(OUT_DIR / 'me_names.json'), 'w', encoding='utf-8') as f:
    json.dump(all_names, f, ensure_ascii=False)
with open(str(OUT_DIR / 'me_countries.json'), 'w', encoding='utf-8') as f:
    json.dump(all_countries, f, ensure_ascii=False)

print(f"Saved {len(loc_feats)} landmarks to {OUT_DIR}", flush=True)

# Also add interpolated points across Iraq
# Create a dense grid of Iraqi points with location embeddings
print("\nAdding dense Iraqi grid points...", flush=True)

# Iraq bounding box: lat 29-37, lon 39-49
# But most cities are in specific areas
IRAQ_CITIES = [
    (32.6164, 44.0323, "Karbala"), (33.3153, 44.3462, "Baghdad"),
    (32.0003, 44.3145, "Najaf"), (30.5086, 47.8144, "Basra"),
    (36.1900, 44.0110, "Erbil"), (36.3400, 43.1300, "Mosul"),
    (35.5564, 45.4352, "Sulaymaniyah"), (34.1983, 43.8744, "Samarra"),
    (31.8356, 47.1457, "Amarah"), (31.0430, 46.2575, "Nasiriyah"),
    (31.9929, 44.9240, "Diwaniyah"), (32.4843, 44.4307, "Hilla"),
    (33.4267, 43.2994, "Ramadi"), (33.3511, 43.7847, "Fallujah"),
    (35.4681, 44.3922, "Kirkuk"), (36.8669, 42.9511, "Duhok"),
    (35.9500, 43.3000, "Erbil2"), (32.5600, 45.2800, "Diwaniyah2"),
    (31.3200, 46.9700, "Amara2"), (33.1000, 44.0100, "Baghdad2"),
]

# Generate intermediate points along roads between cities
grid_points = []
for lat, lon, name in IRAQ_CITIES:
    grid_points.append((lat, lon, name))
    # Add surrounding points (simulate streets/buildings)
    for dlat in [-0.01, 0.01]:
        for dlon in [-0.01, 0.01]:
            grid_points.append((lat + dlat, lon + dlon, f"{name}_near"))
    # Add more distant points
    for dlat in [-0.05, 0.05]:
        for dlon in [-0.05, 0.05]:
            grid_points.append((lat + dlat, lon + dlon, f"{name}_far"))

# Also add points along roads (linear interpolation between cities)
for i in range(len(IRAQ_CITIES)):
    for j in range(i+1, min(i+3, len(IRAQ_CITIES))):
        lat1, lon1, _ = IRAQ_CITIES[i]
        lat2, lon2, _ = IRAQ_CITIES[j]
        for t in np.linspace(0, 1, 10):
            lat = lat1 + t * (lat2 - lat1)
            lon = lon1 + t * (lon2 - lon1)
            grid_points.append((lat, lon, f"road_{i}_{j}_{t:.1f}"))

print(f"Total grid points: {len(grid_points)}", flush=True)

grid_gps = np.array([[p[0], p[1]] for p in grid_points], dtype=np.float32)
grid_tensor = torch.tensor(grid_gps, dtype=torch.float32)

with torch.no_grad():
    grid_feats = gc_model.location_encoder(grid_tensor)
    grid_feats = grid_feats / grid_feats.norm(dim=-1, keepdim=True)
    grid_feats = grid_feats.cpu().numpy().astype(np.float32)

# Combine with landmarks
combined_features = np.concatenate([loc_feats, grid_feats], axis=0)
combined_gps = np.concatenate([gps_arr, grid_gps], axis=0)
combined_names = all_names + [f"grid_{p[2]}" for p in grid_points]

np.save(str(OUT_DIR / 'me_features_dense.npy'), combined_features)
np.save(str(OUT_DIR / 'me_gps_dense.npy'), combined_gps)
with open(str(OUT_DIR / 'me_names_dense.json'), 'w', encoding='utf-8') as f:
    json.dump(combined_names, f, ensure_ascii=False)

lats, lons = combined_gps[:, 0], combined_gps[:, 1]
iraq = ((lons > 39) & (lons < 49) & (lats > 29) & (lats < 37)).sum()
print(f"\nDENSE gallery: {len(combined_features)} points, {iraq} in Iraq box", flush=True)
print("Done!", flush=True)
