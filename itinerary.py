import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict

logger = logging.getLogger(__name__)

class ItineraryRecommender:
    """
    Recommends travel itineraries using collaborative filtering
    """
    def __init__(self, data_path: str = None):
        self.model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self._initialize_data(data_path)
        self._train_model()

    def _initialize_data(self, data_path: str):
        if data_path:
            self.data = pd.read_csv(data_path)
        else:
            self._generate_sample_data()

    def _generate_sample_data(self):
        self.data = pd.DataFrame({
            'user_id': range(1, 101),
            'adventure': np.random.randint(1, 6, 100),
            'culture': np.random.randint(1, 6, 100),
            'budget': np.random.randint(1, 6, 100)
        })

    def _train_model(self):
        try:
            self.model.fit(self.data.drop('user_id', axis=1))
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise

    def recommend(self, preferences: List[float], k: int = 3) -> List[Dict]:
        try:
            distances, indices = self.model.kneighbors([preferences], n_neighbors=k)
            return self.data.iloc[indices[0]].to_dict('records')
        except Exception as e:
            logger.error(f"Recommendation failed: {str(e)}")
            return []
