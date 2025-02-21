import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import List, Dict, Union, Optional
import joblib

logger = logging.getLogger(__name__)

class InvalidPreferenceError(ValueError):
    """Custom exception for invalid preference input"""
    pass

class ModelNotTrainedError(Exception):
    """Custom exception for untrained model usage"""
    pass

class ItineraryRecommender:
    """
    Recommends travel itineraries using collaborative filtering with enhanced features
    
    Attributes:
        model: Trained recommendation pipeline
        scaler: Feature scaler for normalization
        features: List of feature columns used in modeling
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the recommender system
        
        Args:
            data_path: Optional path to CSV data file. If not provided,
                       generates sample data
        """
        self.features = ['adventure', 'culture', 'budget']
        self._initialize_data(data_path)
        self._create_pipeline()
        self._train_model()

    def _initialize_data(self, data_path: Optional[str]):
        """Load or generate data and validate structure"""
        if data_path:
            if not Path(data_path).exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            self.data = pd.read_csv(data_path)
            self._validate_data()
        else:
            self._generate_sample_data()

    def _validate_data(self):
        """Ensure data contains required columns and valid values"""
        missing = set(self.features) - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        if self.data[self.features].isnull().any().any():
            raise ValueError("Data contains missing values in feature columns")

    def _generate_sample_data(self):
        """Generate synthetic sample data with realistic distributions"""
        np.random.seed(42)
        self.data = pd.DataFrame({
            'user_id': range(1, 101),
            'adventure': np.random.normal(3.5, 1.2, 100).clip(1, 5).astype(int),
            'culture': np.random.normal(3.0, 1.5, 100).clip(1, 5).astype(int),
            'budget': np.random.normal(4.0, 0.8, 100).clip(1, 5).astype(int),
            'travel_style': np.random.choice(['solo', 'group'], 100)
        })

    def _create_pipeline(self):
        """Create ML pipeline with preprocessing and model"""
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('nn', NearestNeighbors(
                n_neighbors=15,  # Larger initial pool for flexibility
                metric='cosine',
                algorithm='brute'
            ))
        ])

    def _train_model(self):
        """Train the recommendation model with error handling"""
        try:
            self.pipeline.fit(self.data[self.features])
            self.is_trained = True
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            self.is_trained = False
            raise RuntimeError("Model training failed") from e

    def recommend(
        self,
        preferences: Union[List[float], Dict[str, float]],
        k: int = 3,
        max_distance: float = 0.5
    ) -> List[Dict]:
        """
        Generate travel recommendations based on user preferences
        
        Args:
            preferences: User preferences as list or dict matching feature columns
            k: Maximum number of recommendations to return
            max_distance: Maximum allowed cosine distance (0-1 scale)
            
        Returns:
            List of recommended user profiles with similarity scores
            
        Raises:
            ModelNotTrainedError: If called before training
            InvalidPreferenceError: For invalid input format
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Model must be trained before making recommendations")

        try:
            # Convert dict input to ordered list
            if isinstance(preferences, dict):
                preferences = [preferences.get(f, 3) for f in self.features]
                
            if len(preferences) != len(self.features):
                raise InvalidPreferenceError(
                    f"Expected {len(self.features)} features, got {len(preferences)}"
                )
                
            # Transform and query
            scaled_prefs = self.pipeline['scaler'].transform([preferences])
            distances, indices = self.pipeline['nn'].kneighbors(
                scaled_prefs, 
                n_neighbors=k,
                return_distance=True
            )
            
            # Filter results by distance threshold
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if dist <= max_distance:
                    record = self.data.iloc[idx].to_dict()
                    record['similarity'] = 1 - dist
                    results.append(record)
                    
            return sorted(results, key=lambda x: x['similarity'], reverse=True)
            
        except Exception as e:
            logger.error(f"Recommendation failed: {str(e)}")
            raise InvalidPreferenceError("Invalid preference input") from e

    def save_model(self, path: str):
        """Save trained pipeline to disk"""
        joblib.dump(self.pipeline, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str, data_path: Optional[str] = None):
        """Load trained pipeline from disk"""
        pipeline = joblib.load(path)
        instance = cls(data_path)
        instance.pipeline = pipeline
        instance.is_trained = True
        return instance
