from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, silhouette_score, mean_absolute_error
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd
import plotly.express as px
import numpy as np
from typing import Tuple, List, Dict, Union, Optional
from utils.logger import logger

class ModelTrainer:
    @staticmethod
    def train_injury_model(
        data: pd.DataFrame,
        param_grid: Optional[Dict[str, list]] = None,
        cv: int = 5,
        test_size: float = 0.2
    ) -> Tuple[RandomForestClassifier, pd.DataFrame]:
        """Train and validate an injury prediction model with hyperparameter tuning.

        Args:
            data: DataFrame containing player data and injury risk labels
            param_grid: Custom hyperparameter grid for GridSearchCV. If None, uses default.
            cv: Number of cross-validation folds
            test_size: Proportion of data to use for testing

        Returns:
            Tuple of (best model, validation metrics DataFrame)

        Raises:
            ValueError: If required features are missing or data is empty
        """
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
            estimator=RandomForestClassifier(),
            param_grid=param_grid or default_param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1
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
        
        logger.info(f"Injury model training complete. Best params: {search.best_params_}")
        return best_model, metrics

    @staticmethod
    def train_rating_model(
        data: pd.DataFrame,
        param_dist: Optional[Dict[str, list]] = None,
        n_iter: int = 100,
        test_size: float = 0.2,
        n_jobs: int = -1,
        explain: bool = True
    ) -> Tuple[RandomForestRegressor, pd.DataFrame, Optional[shap.Explanation]]:
        """Train and validate player rating model with advanced features.
        
        Args:
            data: DataFrame containing player performance metrics and ratings
            param_dist: Custom parameter distribution for RandomizedSearchCV
            n_iter: Number of parameter combinations to try
            test_size: Proportion of data to use for testing
            n_jobs: Number of parallel jobs (-1 for all cores)
            explain: Whether to generate SHAP explanations
            
        Returns:
            Tuple containing:
            - Best trained model
            - Validation metrics DataFrame
            - SHAP explanations (if explain=True)
        """
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
        
        # Split data with stratification on binned ratings
        y_bins = pd.qcut(y, q=5, labels=False)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y_bins, random_state=42
        )
        
        # Randomized parameter search with parallel execution
        search = RandomizedSearchCV(
            estimator=RandomForestRegressor(),
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
        
        # Calculate metrics
        metrics = pd.DataFrame({
            'best_params': [search.best_params_],
            'mse': [mean_squared_error(y_test, y_pred)],
            'mae': [mean_absolute_error(y_test, y_pred)],
            'r2': [r2_score(y_test, y_pred)],
            'n_iter': [n_iter],
            'features': [features]
        })
        
        # Generate explanations if requested
        shap_explanation = None
        if explain:
            shap_explanation = ModelTrainer.explain_model_shap(best_model, X_test)
            
            # Log feature impacts
            mean_shap = np.abs(shap_explanation.values).mean(axis=0)
            logger.info("Average SHAP impact:\n" + 
                       pd.Series(mean_shap, index=features)
                       .sort_values(ascending=False)
                       .to_string())
        
        logger.info(f"Rating model training complete. Best params: {search.best_params_}")
        return best_model, metrics, shap_explanation

    @staticmethod
    def cluster_players(
        data: pd.DataFrame,
        features: List[str],
        n_clusters: Optional[int] = None,
        max_clusters: int = 10,
        algorithm: str = 'kmeans',
        scaler: str = 'standard',
        pca_variance: Optional[float] = None
    ) -> Tuple[Union[KMeans, DBSCAN], pd.DataFrame]:
        """Cluster players using various algorithms with automated elbow detection.

        Args:
            data: Input DataFrame containing player metrics
            features: Features to use for clustering
            n_clusters: Fixed number of clusters (None for auto-detection)
            max_clusters: Maximum clusters to consider for elbow method
            algorithm: Clustering algorithm ('kmeans' or 'dbscan')
            scaler: Scaling method ('standard' or 'minmax')
            pca_variance: Optional PCA variance to retain (0.0-1.0)

        Returns:
            Tuple of (trained clustering model, DataFrame with cluster labels)

        Raises:
            ValueError: For invalid algorithm or scaler choices
        """
        ModelTrainer._validate_data(data, features)
        
        # Feature scaling
        scaler = StandardScaler() if scaler == 'standard' else MinMaxScaler()
        scaled_data = scaler.fit_transform(data[features])

        # Dimensionality reduction
        if pca_variance:
            pca = PCA(n_components=pca_variance)
            scaled_data = pca.fit_transform(scaled_data)
            logger.info(f"Reduced to {pca.n_components_} components explaining {pca_variance*100}% variance")

        # Determine optimal clusters
        if n_clusters is None and algorithm == 'kmeans':
            visualizer = KElbowVisualizer(KMeans(), k=(2, max_clusters), metric='distortion')
            visualizer.fit(scaled_data)
            n_clusters = visualizer.elbow_value_
            logger.info(f"Auto-detected optimal clusters: {n_clusters}")

        # Clustering
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        elif algorithm == 'dbscan':
            model = DBSCAN()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Choose 'kmeans' or 'dbscan'")

        clusters = model.fit_predict(scaled_data)
        
        clustered_data = data.copy()
        clustered_data['cluster'] = clusters
        
        # Calculate metrics
        if algorithm == 'kmeans' and n_clusters > 1:
            score = silhouette_score(scaled_data, clusters)
            logger.info(f"Silhouette score: {score:.2f}")
        
        return model, clustered_data

    @staticmethod
    def visualize_elbow(
        data: pd.DataFrame,
        features: List[str],
        k_range: Tuple[int, int] = (2, 12),
        metric: str = 'distortion'
    ) -> None:
        """Visualize elbow method for cluster number selection.

        Args:
            data: Input DataFrame
            features: Features to use for clustering
            k_range: Range of cluster numbers to evaluate
            metric: Elbow metric ('distortion' or 'silhouette')
        """
        scaled_data = StandardScaler().fit_transform(data[features])
        
        visualizer = KElbowVisualizer(
            KMeans(),
            k=k_range,
            metric=metric,
            timings=False
        )
        visualizer.fit(scaled_data)
        visualizer.show()

    @staticmethod
    def visualize_clusters(
        clustered_data: pd.DataFrame,
        features: List[str],
        dimensions: int = 2,
        color_col: str = 'cluster',
        plot_title: str = "Player Clusters"
    ) -> px.scatter:
        """Generate interactive cluster visualization with Plotly.

        Args:
            clustered_data: DataFrame with cluster labels
            features: Original features used for clustering
            dimensions: 2 or 3 for visualization
            color_col: Column to use for coloring points
            plot_title: Custom title for the plot

        Returns:
            Plotly scatter plot object
        """
        pca = PCA(n_components=dimensions)
        reduced_data = pca.fit_transform(clustered_data[features])
        
        plot_data = pd.DataFrame(reduced_data, columns=[f"PC{i+1}" for i in range(dimensions)])
        plot_data[color_col] = clustered_data[color_col].astype(str)
        
        fig = px.scatter(
            plot_data,
            x='PC1',
            y='PC2',
            z='PC3' if dimensions == 3 else None,
            color=color_col,
            title=plot_title,
            hover_data=clustered_data[features]
        ).update_layout(
            width=1200,
            height=800
        )
        
        return fig

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
