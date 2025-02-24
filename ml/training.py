import joblib
import shap
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    silhouette_score
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Dict, Union, Optional
from pathlib import Path
from datetime import datetime
import logging
from utils.logger import logger

class ModelTrainer:
    @staticmethod
    def train_injury_model(
        data: pd.DataFrame,
        param_grid: Optional[Dict[str, list]] = None,
        cv: int = 5,
        test_size: float = 0.2,
        n_jobs: int = -1,
        explain: bool = True
    ) -> Tuple[RandomForestClassifier, pd.DataFrame, Optional[shap.Explanation]]:
        """Train injury prediction model with SHAP explanations."""
        required_features = ['distance_covered', 'sprint_speed', 'tackle_success_rate', 'injury_risk']
        ModelTrainer._validate_data(data, required_features)

        default_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', None]
        }

        X = data[required_features[:-1]]
        y = data['injury_risk']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        search = GridSearchCV(
            RandomForestClassifier(),
            param_grid=param_grid or default_param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=n_jobs
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        val_preds = best_model.predict(X_test)

        metrics = pd.DataFrame({
            'best_params': [search.best_params_],
            'accuracy': [accuracy_score(y_test, val_preds)],
            'f1_score': [f1_score(y_test, val_preds, average='weighted')],
            'cv_folds': [cv]
        })

        shap_explanation = None
        if explain:
            shap_explanation = ModelTrainer.explain_model_shap(best_model, X_test)

        logger.info(f"Injury model training complete")
        return best_model, metrics, shap_explanation

    @staticmethod
    def train_rating_model(
        data: pd.DataFrame,
        param_dist: Optional[Dict[str, list]] = None,
        n_iter: int = 100,
        test_size: float = 0.2,
        n_jobs: int = -1,
        explain: bool = True
    ) -> Tuple[RandomForestRegressor, pd.DataFrame, Optional[shap.Explanation]]:
        """Train player rating model with SHAP explanations."""
        features = ['goals', 'assists', 'pass_accuracy', 'defensive_score', 'xg_contribution']
        ModelTrainer._validate_data(data, features + ['player_rating'])

        default_param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2', 0.8],
            'min_impurity_decrease': [0.0, 0.01, 0.1]
        }

        X = data[features]
        y = data['player_rating']

        y_bins = pd.qcut(y, q=5, labels=False)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y_bins, random_state=42
        )

        search = RandomizedSearchCV(
            RandomForestRegressor(),
            param_distributions=param_dist or default_param_dist,
            n_iter=n_iter,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=n_jobs,
            random_state=42
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        metrics = pd.DataFrame({
            'best_params': [search.best_params_],
            'mse': [mean_squared_error(y_test, y_pred)],
            'mae': [mean_absolute_error(y_test, y_pred)],
            'r2': [r2_score(y_test, y_pred)],
            'n_iter': [n_iter]
        })

        shap_explanation = None
        if explain:
            shap_explanation = ModelTrainer.explain_model_shap(best_model, X_test)

        logger.info(f"Rating model training complete")
        return best_model, metrics, shap_explanation

    @staticmethod
    def cluster_players(
        data: pd.DataFrame,
        features: List[str],
        n_clusters: Optional[int] = None,
        max_clusters: int = 10,
        algorithm: str = 'kmeans',
        scaler: str = 'standard',
        pca_variance: Optional[float] = None,
        outlier_threshold: float = 0.95,
        n_jobs: int = -1
    ) -> Tuple[Union[KMeans, DBSCAN], pd.DataFrame]:
        """Cluster players with outlier detection and parallel processing."""
        ModelTrainer._validate_data(data, features)

        # Feature scaling
        scaler = StandardScaler() if scaler == 'standard' else MinMaxScaler()
        scaled_data = scaler.fit_transform(data[features])

        # Dimensionality reduction
        if pca_variance:
            pca = PCA(n_components=pca_variance)
            scaled_data = pca.fit_transform(scaled_data)
            logger.info(f"Reduced to {pca.n_components_} components")

        # Determine optimal clusters
        if n_clusters is None and algorithm == 'kmeans':
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                visualizer = KElbowVisualizer(KMeans(), k=(2, max_clusters))
                future = executor.submit(visualizer.fit, scaled_data)
                visualizer = future.result()
            n_clusters = visualizer.elbow_value_

        # Clustering
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        elif algorithm == 'dbscan':
            model = DBSCAN(n_jobs=n_jobs)
        else:
            raise ValueError("Unsupported algorithm")

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            clusters = executor.submit(model.fit_predict, scaled_data).result()

        clustered_data = data.copy()
        clustered_data['cluster'] = clusters

        # Outlier detection
        if algorithm == 'kmeans':
            distances = model.transform(scaled_data)
            outlier_scores = np.min(distances, axis=1)
            clustered_data['outlier'] = (outlier_scores > np.quantile(outlier_scores, outlier_threshold)).astype(int)
        elif algorithm == 'dbscan':
            clustered_data['outlier'] = (clusters == -1).astype(int)

        return model, clustered_data

    @staticmethod
    def visualize_feature_importance(
        model: Union[RandomForestClassifier, RandomForestRegressor],
        features: List[str],
        title: str = "Feature Importance"
    ) -> px.bar:
        """Generate interactive feature importance plot."""
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        fig = px.bar(
            importance,
            x='importance',
            y='feature',
            title=title,
            color='importance',
            orientation='h'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        return fig

    @staticmethod
    def explain_model_shap(
        model: Union[RandomForestClassifier, RandomForestRegressor],
        X: pd.DataFrame,
        sample_size: int = 100
    ) -> shap.Explanation:
        """Generate SHAP explanations for model predictions."""
        explainer = shap.TreeExplainer(model)
        sample = X.sample(min(sample_size, len(X)), random_state=42)
        shap_values = explainer.shap_values(sample)
        return shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=sample.values,
            feature_names=X.columns.tolist()
        )

    @staticmethod
    def describe_clusters(
        clustered_data: pd.DataFrame,
        features: List[str]
    ) -> pd.DataFrame:
        """Generate cluster statistics summary."""
        return clustered_data.groupby('cluster')[features].agg(
            ['mean', 'median', 'std', 'min', 'max']
        ).reset_index()

    @staticmethod
    def save_model(
        model: object,
        path: Union[str, Path],
        metadata: Optional[Dict] = None
    ) -> None:
        """Save model with metadata."""
        model_dir = Path(path)
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'model': model,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(save_data, model_dir)
        logger.info(f"Model saved to {model_dir}")

    @staticmethod
    def load_model(
        path: Union[str, Path]
    ) -> Tuple[object, Dict]:
        """Load saved model and metadata."""
        model_dir = Path(path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model file {model_dir} not found")
            
        loaded = joblib.load(model_dir)
        return loaded['model'], loaded['metadata']

    @staticmethod
    def _validate_data(data: pd.DataFrame, required_features: List[str]) -> None:
        """Validate input data meets requirements."""
        if data.empty:
            raise ValueError("Input data cannot be empty")
            
        missing = [feat for feat in required_features if feat not in data.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
            
        if data[required_features].isnull().any().any():
            raise ValueError("Input data contains null values in required features")
