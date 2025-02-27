# itinerary.py
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import List, Dict, Optional, AsyncIterator
import joblib
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, ValidationError, confloat, conint
from config import AsyncConfigManager

logger = logging.getLogger(__name__)

class RecommendationPreferences(BaseModel):
    """Pydantic model for input validation"""
    adventure: conint(ge=1, le=5) = 3
    culture: conint(ge=1, le=5) = 3
    budget: conint(ge=1, le=5) = 3
    max_distance: confloat(ge=0.0, le=1.0) = 0.5
    top_k: conint(ge=1, le=50) = 5

class InvalidPreferenceError(ValueError):
    """Custom exception for invalid preference input"""
    pass

class ModelNotTrainedError(Exception):
    """Custom exception for untrained model usage"""
    pass

class ItineraryRecommender:
    """
    Async-enabled travel itinerary recommender with enhanced features
    
    Implements async context manager pattern for resource management
    """
    
    def __init__(self, config: AsyncConfigManager):
        """
        Initialize the recommender system
        
        Args:
            config: Async configuration manager
        """
        self.config = config
        self.features = ['adventure', 'culture', 'budget']
        self._executor = ThreadPoolExecutor()
        self._pipeline = None
        self.data = None
        self._is_trained = False

    async def __aenter__(self):
        """Async initialization"""
        await self._load_data_async()
        await self._train_model_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources"""
        await self._close_resources()

    async def _load_data_async(self):
        """Async data loading with configurable source"""
        data_path = self.config.settings.get('itinerary_data_path')
        if data_path:
            await self._load_external_data(data_path)
        else:
            await self._generate_sample_data_async()
        self._validate_data()

    async def _load_external_data(self, data_path: str):
        """Load data from external source asynchronously"""
        loop = asyncio.get_running_loop()
        try:
            self.data = await loop.run_in_executor(
                self._executor,
                pd.read_csv,
                data_path
            )
            logger.info(f"Loaded external data from {data_path}")
        except Exception as e:
            logger.error(f"Failed to load external data: {str(e)}")
            raise

    async def _generate_sample_data_async(self):
        """Generate synthetic sample data asynchronously"""
        loop = asyncio.get_running_loop()
        try:
            self.data = await loop.run_in_executor(
                self._executor,
                self._create_sample_dataframe
            )
            logger.info("Generated synthetic sample data")
        except Exception as e:
            logger.error(f"Sample data generation failed: {str(e)}")
            raise

    def _create_sample_dataframe(self) -> pd.DataFrame:
        """Create sample data (runs in executor)"""
        np.random.seed(self.config.settings.social_algorithm_params.get('random_state', 42))
        return pd.DataFrame({
            'user_id': range(1, 1001),
            'adventure': np.random.normal(3.5, 1.2, 1000).clip(1, 5).astype(int),
            'culture': np.random.normal(3.0, 1.5, 1000).clip(1, 5).astype(int),
            'budget': np.random.normal(4.0, 0.8, 1000).clip(1, 5).astype(int),
            'travel_style': np.random.choice(['solo', 'group'], 1000)
        })

    def _validate_data(self):
        """Validate data structure and quality"""
        required_columns = set(self.features + ['user_id'])
        missing = required_columns - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        if self.data[self.features].isnull().any().any():
            raise ValueError("Data contains missing values in feature columns")

    async def _train_model_async(self):
        """Async model training with progress tracking"""
        loop = asyncio.get_running_loop()
        try:
            self._pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('nn', NearestNeighbors(
                    n_neighbors=self.config.settings.recommendation_top_k * 3,
                    metric='cosine',
                    algorithm='brute'
                ))
            ])
            
            await loop.run_in_executor(
                self._executor,
                self._pipeline.fit,
                self.data[self.features]
            )
            self._is_trained = True
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            self._is_trained = False
            raise

    async def recommend(
        self,
        preferences: Dict[str, int],
        max_distance: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Generate travel recommendations with async support
        
        Args:
            preferences: User preferences dictionary
            max_distance: Maximum allowed cosine distance
            top_k: Maximum number of recommendations
            
        Returns:
            List of recommended user profiles with similarity scores
        """
        try:
            validated = RecommendationPreferences(
                **preferences,
                max_distance=max_distance or self.config.settings.recommendation_max_distance,
                top_k=top_k or self.config.settings.recommendation_top_k
            )
        except ValidationError as e:
            raise InvalidPreferenceError(f"Invalid preferences: {str(e)}") from e

        if not self._is_trained:
            raise ModelNotTrainedError("Model must be trained before recommendations")

        loop = asyncio.get_running_loop()
        try:
            prefs_list = [validated.adventure, validated.culture, validated.budget]
            
            scaled_prefs = self._pipeline['scaler'].transform([prefs_list])
            
            distances, indices = await loop.run_in_executor(
                self._executor,
                lambda: self._pipeline['nn'].kneighbors(
                    scaled_prefs, 
                    n_neighbors=validated.top_k,
                    return_distance=True
                )
            )
            
            return await self._process_results(
                distances[0],
                indices[0],
                validated.max_distance
            )
        except Exception as e:
            logger.error(f"Recommendation failed: {str(e)}")
            raise InvalidPreferenceError("Recommendation processing error") from e

    async def _process_results(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        max_distance: float
    ) -> List[Dict]:
        """Process and filter results asynchronously"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_process_results,
            distances,
            indices,
            max_distance
        )

    def _sync_process_results(self, distances, indices, max_distance):
        """CPU-bound result processing (runs in executor)"""
        results = []
        for dist, idx in zip(distances, indices):
            if dist <= max_distance:
                record = self.data.iloc[idx].to_dict()
                record['similarity'] = 1 - dist
                results.append(record)
        return sorted(results, key=lambda x: x['similarity'], reverse=True)

    async def _close_resources(self):
        """Cleanup thread pool and resources"""
        self._executor.shutdown(wait=False)
        logger.debug("Thread pool executor shutdown")

    async def save_model_async(self, path: str):
        """Async model persistence"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            joblib.dump,
            self._pipeline,
            path
        )
        logger.info(f"Model saved to {path}")

    @classmethod
    async def load_model_async(
        cls,
        config: AsyncConfigManager,
        path: str
    ) -> 'ItineraryRecommender':
        """Async model loading"""
        loop = asyncio.get_running_loop()
        pipeline = await loop.run_in_executor(
            self._executor,
            joblib.load,
            path
        )
        instance = cls(config)
        instance._pipeline = pipeline
        instance._is_trained = True
        return instance
