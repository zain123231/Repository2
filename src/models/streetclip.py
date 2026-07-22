"""
StreetCLIP Zero-Shot — hierarchical text-based geolocation.
Uses CLIP text-image alignment with city/country templates.
"""
import numpy as np
import torch
import clip


CITIES_45 = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "London", "Paris", "Berlin", "Madrid", "Rome",
    "Tokyo", "Beijing", "Shanghai", "Mumbai", "Delhi",
    "Cairo", "Istanbul", "Dubai", "Riyadh", "Baghdad",
    "Moscow", "São Paulo", "Buenos Aires", "Mexico City", "Lima",
    "Sydney", "Melbourne", "Toronto", "Vancouver", "Singapore",
    "Bangkok", "Seoul", "Taipei", "Jakarta", "Kuala Lumpur",
    "Nairobi", "Lagos", "Casablanca", "Cape Town", "Johannesburg",
    "Amsterdam", "Vienna", "Prague", "Warsaw", "Athens"
]

COUNTRIES_50 = [
    "United States", "United Kingdom", "France", "Germany", "Spain",
    "Italy", "Japan", "China", "India", "Egypt",
    "Turkey", "United Arab Emirates", "Saudi Arabia", "Iraq", "Russia",
    "Brazil", "Argentina", "Mexico", "Peru", "Australia",
    "Canada", "Singapore", "Thailand", "South Korea", "Taiwan",
    "Indonesia", "Malaysia", "Kenya", "Nigeria", "Morocco",
    "South Africa", "Netherlands", "Austria", "Czech Republic", "Poland",
    "Greece", "Sweden", "Norway", "Denmark", "Finland",
    "Switzerland", "Belgium", "Portugal", "Ireland", "New Zealand",
    "Colombia", "Chile", "Philippines", "Vietnam", "Pakistan"
]


class StreetCLIPZeroShot:
    """
    Zero-shot geolocation using CLIP text-image matching.
    Template: "a street photo taken in {city}, {country}"
    """
    def __init__(self, clip_extractor, device="cpu"):
        self.clip = clip_extractor
        self.device = device
        self.city_features = None
        self.country_features = None
        self._build_text_features()

    def _build_text_features(self):
        """Pre-encode all city and country text prompts."""
        city_prompts = [f"a street photo taken in {city}" for city in CITIES_45]
        country_prompts = [f"a street photo taken in {country}" for country in COUNTRIES_50]

        self.city_features = self.clip.encode_texts(city_prompts)
        self.country_features = self.clip.encode_texts(country_prompts)
        self.city_names = CITIES_45
        self.country_names = COUNTRIES_50

        print(f"StreetCLIP: encoded {len(CITIES_45)} cities, {len(COUNTRIES_50)} countries")

    def predict_city(self, images):
        """Predict most likely city for each image."""
        image_features = self.clip.extract(images)
        similarities = image_features @ self.city_features.T
        city_indices = np.argmax(similarities, axis=1)
        scores = np.max(similarities, axis=1)
        return city_indices, scores

    def predict_country(self, images):
        """Predict most likely country for each image."""
        image_features = self.clip.extract(images)
        similarities = image_features @ self.country_features.T
        country_indices = np.argmax(similarities, axis=1)
        scores = np.max(similarities, axis=1)
        return country_indices, scores

    def predict_hierarchical(self, images):
        """
        Two-stage prediction:
        1. Predict country
        2. Within predicted country, predict city
        Returns: (country_name, city_name, confidence)
        """
        country_idx, country_score = self.predict_country(images)
        results = []
        for i, idx in enumerate(country_idx):
            country = self.country_names[idx]
            results.append((country, None, float(country_score[i])))
        return results


class CityCoordinateMapper:
    """Maps city names to approximate coordinates."""
    CITY_COORDS = {
        "New York": (40.7128, -74.0060),
        "Los Angeles": (34.0522, -118.2437),
        "Chicago": (41.8781, -87.6298),
        "Houston": (29.7604, -95.3698),
        "Phoenix": (33.4484, -112.0740),
        "London": (51.5074, -0.1278),
        "Paris": (48.8566, 2.3522),
        "Berlin": (52.5200, 13.4050),
        "Madrid": (40.4168, -3.7038),
        "Rome": (41.9028, 12.4964),
        "Tokyo": (35.6762, 139.6503),
        "Beijing": (39.9042, 116.4074),
        "Shanghai": (31.2304, 121.4737),
        "Mumbai": (19.0760, 72.8777),
        "Delhi": (28.7041, 77.1025),
        "Cairo": (30.0444, 31.2357),
        "Istanbul": (41.0082, 28.9784),
        "Dubai": (25.2048, 55.2708),
        "Riyadh": (24.7136, 46.6753),
        "Baghdad": (33.3152, 44.3661),
        "Moscow": (55.7558, 37.6173),
        "São Paulo": (-23.5505, -46.6333),
        "Buenos Aires": (-34.6037, -58.3816),
        "Mexico City": (19.4326, -99.1332),
        "Lima": (-12.0464, -77.0428),
        "Sydney": (-33.8688, 151.2093),
        "Melbourne": (-37.8136, 144.9631),
        "Toronto": (43.6532, -79.3832),
        "Vancouver": (49.2827, -123.1207),
        "Singapore": (1.3521, 103.8198),
        "Bangkok": (13.7563, 100.5018),
        "Seoul": (37.5665, 126.9780),
        "Taipei": (25.0330, 121.5654),
        "Jakarta": (-6.2088, 106.8456),
        "Kuala Lumpur": (3.1390, 101.6869),
        "Nairobi": (-1.2921, 36.8219),
        "Lagos": (6.5244, 3.3792),
        "Casablanca": (33.5731, -7.5898),
        "Cape Town": (-33.9249, 18.4241),
        "Johannesburg": (-26.2041, 28.0473),
        "Amsterdam": (52.3676, 4.9041),
        "Vienna": (48.2082, 16.3738),
        "Prague": (50.0755, 14.4378),
        "Warsaw": (52.2297, 21.0122),
        "Athens": (37.9838, 23.7275),
    }

    @classmethod
    def get_coordinate(cls, city_name):
        return cls.CITY_COORDS.get(city_name, (0.0, 0.0))

    @classmethod
    def get_all_coordinates(cls):
        return np.array([cls.CITY_COORDS[c] for c in CITIES_45 if c in cls.CITY_COORDS])
