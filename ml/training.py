from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
from typing import Tuple, List
from utils.logger import logger

class ModelTrainer:
    @staticmethod
    def train_injury_model(data: pd.DataFrame) -> RandomForestClassifier:
        """Train injury prediction model with enhanced validation"""
        required_features = ['distance_covered', 'sprint_speed', 'tackle_success_rate', 'injury_risk']
        ModelTrainer._validate_data(data, required_features)
        
        X = data[required_features[:-1]]
        y = data['injury_risk']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'class_weight': ['balanced', None]
        }
        
        model = GridSearchCV(
            RandomForestClassifier(),
            param_grid=param_grid,
            cv=5,
            scoring='f1_weighted'
        )
        model.fit(X_train, y_train)
        
        logger.info(f"Best injury model params: {model.best_params_}")
        logger.info(f"Validation accuracy: {model.score(X_test, y_test):.2f}")
        
        return model.best_estimator_

    @staticmethod
    def train_rating_model(data: pd.DataFrame) -> RandomForestRegressor:
        """Train player rating model with feature importance analysis"""
        features = ['goals', 'assists', 'pass_accuracy', 'defensive_score', 'xg_contribution']
        ModelTrainer._validate_data(data, features + ['player_rating'])
        
        X = data[features]
        y = data['player_rating']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Model MSE: {mse:.2f}")
        
        # Log feature importances
        importances = pd.Series(model.feature_importances_, index=features)
        logger.info("Feature importances:\n" + importances.sort_values(ascending=False).to_string())
        
        return model

    @staticmethod
    def cluster_players(data: pd.DataFrame, 
                      features: List[str],
                      n_clusters: int = 5) -> Tuple[KMeans, pd.DataFrame]:
        """Cluster players based on performance metrics
        
        Args:
            data: DataFrame containing player data
            features: List of features to use for clustering
            n_clusters: Number of clusters to create
            
        Returns:
            Tuple containing trained KMeans model and data with cluster labels
        """
        ModelTrainer._validate_data(data, features)
        
        # Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[features])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to data
        clustered_data = data.copy()
        clustered_data['cluster'] = clusters
        
        # Calculate silhouette score
        score = silhouette_score(scaled_data, clusters)
        logger.info(f"Clustering silhouette score: {score:.2f}")
        
        return kmeans, clustered_data

    @staticmethod
    def visualize_clusters(clustered_data: pd.DataFrame,
                         features: List[str],
                         dimensions: int = 2) -> None:
        """Visualize player clusters using Plotly
        
        Args:
            clustered_data: Data with cluster labels
            features: Features used for clustering
            dimensions: 2D or 3D visualization (2 or 3)
        """
        if dimensions not in [2, 3]:
            raise ValueError("Dimensions must be 2 or 3")
            
        # Reduce dimensionality
        pca = PCA(n_components=dimensions)
        reduced_data = pca.fit_transform(clustered_data[features])
        
        # Create plot
        plot_data = pd.DataFrame(reduced_data, columns=[f"PC{i+1}" for i in range(dimensions)])
        plot_data['cluster'] = clustered_data['cluster'].astype(str)
        
        if dimensions == 2:
            fig = px.scatter(plot_data, x='PC1', y='PC2', color='cluster',
                           title="Player Clusters (2D PCA Projection)")
        else:
            fig = px.scatter_3d(plot_data, x='PC1', y='PC2', z='PC3', color='cluster',
                               title="Player Clusters (3D PCA Projection)")
        
        fig.show()

    @staticmethod
    def _validate_data(data: pd.DataFrame, required_features: List[str]) -> None:
        """Validate input data contains required features"""
        missing = [feat for feat in required_features if feat not in data.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
            
        if data.empty:
            raise ValueError("Input data cannot be empty")
