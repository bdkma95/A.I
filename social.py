# social.py
import os
import logging
from typing import List
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

class SocialConnector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.db = self._init_firebase()
    
    def _init_firebase(self):
        try:
            if not firebase_admin._apps:
                cred_path = os.getenv("FIREBASE_CRED_PATH")
                if cred_path is None:
                    raise ValueError("FIREBASE_CRED_PATH not set")
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
            return firestore.client()
        except Exception as e:
            logger.error(f"Firebase init failed: {str(e)}")
            return None
    
    def match_travelers(self, interests: List[str], n_clusters: int = 3) -> dict:
        try:
            X = self.vectorizer.fit_transform(interests)
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(X)
            return {
                "labels": kmeans.labels_.tolist(),
                "centroids": kmeans.cluster_centers_.tolist()
            }
        except Exception as e:
            logger.error(f"Matching failed: {str(e)}")
            return {}
