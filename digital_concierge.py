import pandas as pd
from sklearn.neighbors import NearestNeighbors
import spacy
import requests
from bs4 import BeautifulSoup
import geopandas as gpd
from transformers import pipeline
from googletrans import Translator
import pytesseract
import cv2

# Load NLP model for preference parsing
nlp = spacy.load("en_core_web_sm")

# Scrape travel blogs to build a dataset
def scrape_travel_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    attractions = [h2.text for h2 in soup.find_all('h2', class_='attraction')]
    return attractions

# Collaborative Filtering Recommender
class ItineraryRecommender:
    def __init__(self):
        self.data = pd.DataFrame({
            'user': [1, 2, 3],
            'adventure': [5, 3, 4],
            'culture': [2, 5, 3],
            'budget': [4, 2, 5]
        })
        self.model = NearestNeighbors(n_neighbors=2)
        self.model.fit(self.data.drop('user', axis=1))
    
    def recommend(self, preferences):
        _, indices = self.model.kneighbors([preferences])
        return self.data.iloc[indices[0]]

# Fetch safety data from SafetiPin API
def get_safety_data(lat, lon):
    response = requests.get(f"https://api.safetipin.com/safe?lat={lat}&lon={lon}")
    return response.json()

# Sentiment analysis for news/social media
def analyze_risk(text):
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return classifier(text)

# Geospatial risk classification
risk_zones = gpd.read_file("crime_data.geojson")
high_risk = risk_zones[risk_zones['crime_rate'] > 0.7]
print(high_risk.plot())

# Real-time text translation
def translate_text(text, dest_lang='es'):
    translator = Translator()
    return translator.translate(text, dest=dest_lang).text

# OCR for sign translation
def translate_image(img_path):
    img = cv2.imread(img_path)
    text = pytesseract.image_to_string(img)
    return translate_text(text)
