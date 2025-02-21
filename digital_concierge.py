import pandas as pd
from sklearn.neighbors import NearestNeighbors
import spacy
import requests
from bs4 import BeautifulSoup

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
