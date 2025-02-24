from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from utils.logger import logger

class ModelTrainer:
    @staticmethod
    def train_injury_model(data: pd.DataFrame):
        """Train injury prediction model"""
        X = data[['distance_covered', 'sprint_speed', 'tackle_success_rate']]
        y = data['injury_risk']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = GridSearchCV(RandomForestClassifier(), param_grid={'n_estimators': [100, 200]}, cv=5)
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def train_rating_model(data: pd.DataFrame):
        """Train player rating model"""
        features = ['goals', 'assists', 'pass_accuracy', 'defensive_score']
        X = data[features]
        y = data['player_rating']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        logger.info(f"Model MSE: {mean_squared_error(y_test, model.predict(X_test)):.2f}")
        return model
