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
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import firebase_admin
from firebase_admin import firestore
from selenium import webdriver
from sklearn.linear_model import LinearRegression
import speech_recognition as sr
from openweathermap import OpenWeatherMap

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

# Match travelers by interests
def match_travelers(interests):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(interests)
    kmeans = KMeans(n_clusters=3).fit(X)
    return kmeans.labels_

# Firebase integration for real-time chat
def init_firebase():
    cred = firebase_admin.credentials.Certificate("path/to/credentials.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    return db

# Web scraping flight prices
def scrape_flights(url):
    driver = webdriver.Chrome()
    driver.get(url)
    prices = driver.find_elements_by_class_name('flight-price')
    return [float(price.text.replace('$', '')) for price in prices]

# Predictive pricing model
def train_price_model(data):
    X = pd.DataFrame(data['days_in_advance'])
    y = data['price']
    model = LinearRegression().fit(X, y)
    return model.predict([[30]])  # Predict price 30 days ahead

# Weather alerts
def get_weather_alerts(lat, lon):
    owm = OpenWeatherMap(api_key="YOUR_KEY")
    return owm.get_weather_alerts(lat, lon)

# Voice SOS command
def emergency_listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        if "SOS" in text.upper():
            send_emergency_alert()
    except:
        pass
