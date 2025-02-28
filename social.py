# social.py
import logging
import numpy as np
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field, ValidationError
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.pipeline import Pipeline
from config import AsyncConfigManager
from exceptions import SocialConnectionError, ClusteringError
import firebase_admin
from firebase_admin import firestore, credentials
from nltk.corpus import stopwords
import joblib

logger = logging.getLogger(__name__)

class ClusterRequest(BaseModel):
    """Pydantic model for clustering request validation"""
    interests: List[str] = Field(..., min_items=2, max_items=1000)
    algorithm: str = Field('kmeans', regex='^(kmeans|dbscan)$')
    params: Dict[str, Any] = Field(default_factory=dict)

class ClusterResult(BaseModel):
    """Structured clustering result model"""
    labels: List[int]
    centroids: Optional[List[List[float]]]
    top_terms: Dict[int, List[str]]
    quality_metrics: Dict[str, float]
    cluster_distribution: Dict[int, int]
    algorithm: str
    feature_dimensions: int

class SocialConnector:
    """
    Async social connector with advanced clustering and Firebase integration
    
    Features:
    - Async context manager pattern
    - Configurable clustering algorithms
    - Firebase Firestore integration
    - Comprehensive NLP preprocessing
    - Quality metrics and diagnostics
    - Model versioning and caching
    """
    
    def __init__(self, config: AsyncConfigManager):
        self.config = config
        self._executor = ThreadPoolExecutor()
        self._vectorizer = None
        self._pipeline = None
        self._db = None
        self._model_cache = TTLCache(maxsize=10, ttl=timedelta(hours=1))

    async def __aenter__(self):
        """Async initialization"""
        await self._initialize_firebase_async()
        self._initialize_nlp_pipeline()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources"""
        await self._close_firebase_async()
        self._executor.shutdown(wait=False)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    async def _initialize_firebase_async(self):
        """Async Firebase initialization with credentials validation"""
        try:
            if not firebase_admin._apps:
                cred_path = Path(self.config.settings.firebase_cred_path)
                
                if not await self._check_file_exists(cred_path):
                    raise FileNotFoundError(f"Firebase credential file not found: {cred_path}")
                    
                loop = asyncio.get_running_loop()
                cred = await loop.run_in_executor(
                    self._executor,
                    credentials.Certificate,
                    cred_path
                )
                await loop.run_in_executor(
                    self._executor,
                    firebase_admin.initialize_app,
                    cred
                )
                
            self._db = firestore.AsyncClient()
            logger.info("Firebase connection established")

        except Exception as e:
            logger.critical(f"Firebase initialization failed: {str(e)}")
            raise SocialConnectionError("Firebase connection failed") from e

    async def _close_firebase_async(self):
        """Cleanup Firebase resources"""
        if self._db:
            await self._db.close()
        logger.debug("Firebase connection closed")

    def _initialize_nlp_pipeline(self):
        """Initialize NLP components from config"""
        self._vectorizer = TfidfVectorizer(
            stop_words=set(stopwords.words('english')),
            max_features=self.config.settings.social_max_features,
            ngram_range=(1, 2),
            analyzer=self._lemma_analyzer
        )

    def _lemma_analyzer(self, text: str) -> List[str]:
        """Custom analyzer with lemmatization using config NLP"""
        return [
            self.config.nlp.vocab[token.lemma_].text 
            for token in self.config.nlp(text)
            if not token.is_stop and token.is_alpha
        ]

    async def match_travelers(self, request: ClusterRequest) -> ClusterResult:
        """
        Perform async traveler clustering with advanced metrics
        
        Args:
            request: ClusterRequest with parameters
            
        Returns:
            ClusterResult with detailed analysis
        """
        try:
            validated = self._validate_request(request)
            cache_key = self._generate_cache_key(validated)
            
            if cached := self._model_cache.get(cache_key):
                return cached
                
            X = await self._process_texts_async(validated.interests)
            labels = await self._perform_clustering_async(X, validated)
            result = await self._analyze_clusters_async(X, labels, validated)
            
            self._model_cache[cache_key] = result
            return result
            
        except ValidationError as e:
            raise ClusteringError(f"Invalid request: {str(e)}") from e
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            raise ClusteringError("Traveler matching failed") from e

    async def _process_texts_async(self, texts: List[str]) -> Any:
        """Async text processing and vectorization"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._vectorizer.fit_transform,
            texts
        )

    async def _perform_clustering_async(self, X, request: ClusterRequest) -> np.ndarray:
        """Async clustering execution"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._cluster_batch,
            X,
            request
        )

    def _cluster_batch(self, X, request: ClusterRequest) -> np.ndarray:
        """Synchronous clustering implementation"""
        algorithm = request.algorithm
        params = request.params or self.config.settings.social_algorithm_params
        
        if algorithm == 'kmeans':
            model = KMeans(
                n_clusters=params.get('n_clusters', 5),
                random_state=params.get('random_state', 42),
                n_init='auto'
            )
        elif algorithm == 'dbscan':
            model = DBSCAN(
                eps=params.get('eps', 0.5),
                min_samples=params.get('min_samples', 5)
            )
            
        return model.fit_predict(X)

    async def _analyze_clusters_async(self, X, labels: np.ndarray, request: ClusterRequest) -> ClusterResult:
        """Comprehensive cluster analysis"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_analyze_clusters,
            X,
            labels,
            request
        )

    def _sync_analyze_clusters(self, X, labels: np.ndarray, request: ClusterRequest) -> ClusterResult:
        """Synchronous cluster analysis"""
        unique_labels = np.unique(labels[labels != -1])
        feature_names = self._vectorizer.get_feature_names_out()
        
        result = ClusterResult(
            labels=labels.tolist(),
            centroids=[],
            top_terms={},
            quality_metrics={},
            cluster_distribution=dict(zip(*np.unique(labels, return_counts=True))),
            algorithm=request.algorithm,
            feature_dimensions=X.shape[1]
        )
        
        # Calculate quality metrics
        if len(unique_labels) > 1:
            result.quality_metrics['silhouette'] = silhouette_score(X, labels)
            result.quality_metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        
        # Extract cluster centroids and top terms
        if request.algorithm == 'kmeans':
            result.centroids = self._pipeline.named_steps['clusterer'].cluster_centers_.tolist()
            
        for label in unique_labels:
            mask = labels == label
            cluster_texts = X[mask]
            
            # Get top terms using mean TF-IDF scores
            if cluster_texts.shape[0] > 0:
                avg_scores = np.asarray(cluster_texts.mean(axis=0)).ravel()
                top_indices = avg_scores.argsort()[-10:][::-1]
                result.top_terms[int(label)] = [
                    feature_names[i] for i in top_indices 
                    if avg_scores[i] > 0.01
                ]
                
        return result

    async def save_model_async(self, path: str) -> None:
        """Async model persistence"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            joblib.dump,
            self._pipeline,
            path
        )
        logger.info(f"Model saved to {path}")

    async def load_model_async(self, path: str) -> None:
        """Async model loading"""
        loop = asyncio.get_running_loop()
        self._pipeline = await loop.run_in_executor(
            self._executor,
            joblib.load,
            path
        )
        logger.info(f"Model loaded from {path}")

    def _validate_request(self, request: ClusterRequest) -> ClusterRequest:
        """Validate and normalize request parameters"""
        if request.algorithm == 'kmeans' and 'n_clusters' not in request.params:
            request.params['n_clusters'] = self.config.settings.social_default_clusters
        return request

    def _generate_cache_key(self, request: ClusterRequest) -> str:
        """Generate unique cache key for clustering requests"""
        return f"{request.algorithm}_{hash(frozenset(request.interests))}"

    async def _check_file_exists(self, path: Path) -> bool:
        """Async file existence check"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            path.exists
        )
