# social.py
import os
import logging
from typing import List, Dict, Optional, TypedDict
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import joblib

logger = logging.getLogger(__name__)

class ClusterResult(TypedDict):
    labels: List[int]
    centroids: Optional[List[List[float]]]
    top_terms: Dict[int, List[str]]
    silhouette_score: Optional[float]
    cluster_counts: Dict[int, int]

class FirebaseConnectionError(Exception):
    """Custom exception for Firebase connection failures"""

class SocialConnector:
    """
    Enhanced traveler matching system with Firebase integration and advanced clustering
    
    Features:
    - Multiple clustering algorithms
    - Text preprocessing with lemmatization
    - Cluster interpretation with top terms
    - Model persistence
    - Performance metrics
    """
    
    def __init__(self, use_lemmatization: bool = True):
        self._initialize_firebase()
        self.vectorizer = TfidfVectorizer(
            stop_words=set(stopwords.words('english')),
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('clusterer', KMeans(n_init=10, random_state=42))
        ])

    def _initialize_firebase(self) -> None:
        """Initialize Firebase connection with validation"""
        try:
            if not firebase_admin._apps:
                cred_path = Path(os.getenv("FIREBASE_CRED_PATH", ""))
                
                if not cred_path.exists():
                    raise FileNotFoundError(f"Credential file not found: {cred_path}")
                    
                if cred_path.suffix != '.json':
                    raise ValueError("Invalid credential file format")
                    
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                
            self.db = firestore.client()
            logger.info("Firebase connection established")
            
        except Exception as e:
            logger.critical(f"Firebase initialization failed: {str(e)}")
            raise FirebaseConnectionError("Firebase connection failed") from e

    def _preprocess_text(self, texts: List[str]) -> List[str]:
        """Clean and normalize input text"""
        processed = []
        for text in texts:
            # Basic cleaning
            text = text.lower().strip()
            # Lemmatization if enabled
            if self.lemmatizer:
                text = ' '.join([self.lemmatizer.lemmatize(word) 
                               for word in text.split()])
            processed.append(text)
        return processed

    def match_travelers(
        self,
        interests: List[str],
        algorithm: str = 'kmeans',
        n_clusters: Optional[int] = None
    ) -> ClusterResult:
        """
        Cluster travelers by interests with multiple algorithm support
        
        Args:
            interests: List of interest strings
            algorithm: Clustering algorithm ('kmeans' or 'dbscan')
            n_clusters: Optional number of clusters (for KMeans)
            
        Returns:
            ClusterResult with detailed clustering information
        """
        try:
            if not interests or len(interests) < 2:
                raise ValueError("At least 2 interest entries required")
                
            processed = self._preprocess_text(interests)
            X = self.vectorizer.fit_transform(processed)
            
            if algorithm == 'dbscan':
                self.pipeline.steps[-1] = ('clusterer', DBSCAN())
            elif n_clusters:
                self.pipeline.set_params(clusterer__n_clusters=n_clusters)
                
            labels = self.pipeline.fit_predict(X)
            
            return self._format_result(X, labels, processed)
            
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            return ClusterResult(
                labels=[],
                centroids=None,
                top_terms={},
                silhouette_score=None,
                cluster_counts={}
            )

    def _format_result(self, X, labels: np.ndarray, texts: List[str]) -> ClusterResult:
        """Format clustering results with metadata"""
        unique_labels = set(labels)
        result: ClusterResult = {
            'labels': labels.tolist(),
            'centroids': None,
            'top_terms': {},
            'silhouette_score': None,
            'cluster_counts': {label: int(np.sum(labels == label)) 
                              for label in unique_labels}
        }
        
        # Calculate metrics for partition-based clustering
        if isinstance(self.pipeline.named_steps['clusterer'], KMeans):
            result['centroids'] = self.pipeline['clusterer'].cluster_centers_.tolist()
            result['silhouette_score'] = silhouette_score(X, labels)
            
        # Extract top terms for each cluster
        feature_names = self.vectorizer.get_feature_names_out()
        for label in unique_labels:
            if label == -1:  # Skip noise for DBSCAN
                continue
                
            mask = labels == label
            cluster_texts = ' '.join([texts[i] for i in np.where(mask)[0]])
            tfidf_scores = np.mean(X[mask], axis=0).A1
            top_indices = tfidf_scores.argsort()[-5:][::-1]
            
            result['top_terms'][int(label)] = [
                feature_names[i] for i in top_indices
            ]
            
        return result

    def save_model(self, path: str) -> None:
        """Save trained clustering pipeline to disk"""
        joblib.dump(self.pipeline, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load pre-trained clustering pipeline"""
        self.pipeline = joblib.load(path)
        logger.info("Model loaded successfully")
