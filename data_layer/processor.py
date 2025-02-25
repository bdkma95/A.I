import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from functools import lru_cache
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pandarallel import pandarallel  # For parallel processing

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'imputation_strategy': 'mean',
            'scale_features': True,
            'feature_params': {
                'pass_accuracy_threshold': 0.3,
                'distance_ratio_bins': 5
            }
        }
        pandarallel.initialize(progress_bar=False)  # Initialize parallel processing
        
        # Initialize sklearn components
        self.imputer = SimpleImputer(strategy=self.config['imputation_strategy'])
        self.scaler = StandardScaler()

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and transform raw football data into analytical features.
        
        Args:
            data: Raw input dataframe containing player/match statistics
            
        Returns:
            Processed DataFrame with engineered features and clean data
            
        Raises:
            ValueError: If required columns are missing
        """
        try:
            # Validate input data
            self._validate_input(data)
            
            # Create copy to avoid modifying original data
            processed_data = data.copy()
            
            # Handle missing values
            processed_data = self._handle_missing_values(processed_data)
            
            # Create base features
            processed_data = self._create_base_features(processed_data)
            
            # Create advanced metrics
            processed_data = self._create_advanced_metrics(processed_data)
            
            # Create temporal features
            if 'match_date' in processed_data.columns:
                processed_data = self._create_temporal_features(processed_data)
            
            # Normalize features
            if self.config['scale_features']:
                processed_data = self._scale_features(processed_data)
                
            return processed_data
            
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            raise

    def _validate_input(self, data: pd.DataFrame):
        """Validate input dataframe structure"""
        required_columns = {
            'expected_goals', 'expected_assists',
            'successful_passes', 'total_passes'
        }
        
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with configured strategy"""
        # Custom imputation for specific columns
        if 'distance_covered' in data.columns:
            data['distance_covered'] = data['distance_covered'].fillna(
                data['distance_covered'].median()
            )
            
        # General imputation
        numeric_cols = data.select_dtypes(include=np.number).columns
        data[numeric_cols] = self.imputer.fit_transform(data[numeric_cols])
        
        return data

    def _create_base_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create fundamental performance metrics"""
        # Basic features
        data['xg_contribution'] = data['expected_goals'] + data['expected_assists']
        data['pass_accuracy'] = (
            data['successful_passes'] / data['total_passes']
        ).clip(lower=self.config['feature_params']['pass_accuracy_threshold'])
        
        # Positional efficiency
        if {'touches', 'possession_lost'}.issubset(data.columns):
            data['possession_efficiency'] = (
                data['touches'] / (data['possession_lost'] + 1)
            )
            
        return data

    def _create_advanced_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create complex performance indicators"""
        # Defensive contributions
        if {'tackles', 'interceptions'}.issubset(data.columns):
            data['defensive_impact'] = (
                0.7 * data['tackles'] + 0.3 * data['interceptions']
            )
            
        # Physical performance ratios
        if {'distance_covered', 'minutes_played'}.issubset(data.columns):
            data['distance_per_min'] = (
                data['distance_covered'] / data['minutes_played']
            )
            data['distance_ratio'] = pd.qcut(
                data['distance_per_min'],
                self.config['feature_params']['distance_ratio_bins'],
                labels=False
            )
            
        # Create interaction features
        if all(col in data.columns for col in ['xg_contribution', 'pass_accuracy']):
            data['xg_pass_ratio'] = data['xg_contribution'] * data['pass_accuracy']
            
        return data

    def _create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based rolling features"""
        if 'player_id' in data.columns:
            rolling_window = self.config.get('rolling_window', 5)
            
            data = data.sort_values(['player_id', 'match_date'])
            
            for col in ['xg_contribution', 'pass_accuracy']:
                data[f'{col}_rolling_avg'] = (
                    data.groupby('player_id')[col]
                    .transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())
                )
                
        return data

    def _scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features"""
        numeric_cols = data.select_dtypes(include=np.number).columns
        scaled_features = self.scaler.fit_transform(data[numeric_cols])
        data[numeric_cols] = scaled_features
        return data

    @lru_cache(maxsize=128)
    def get_feature_metadata(self) -> Dict:
        """Get metadata about processed features"""
        return {
            'xg_contribution': 'Sum of expected goals and assists',
            'pass_accuracy': 'Ratio of successful passes to total passes',
            'defensive_impact': 'Weighted combination of tackles and interceptions',
            'distance_per_min': 'Distance covered per minute played'
        }

    def batch_process(self, data_iter: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Process multiple batches of data in parallel"""
        return [self.process_data(batch) for batch in data_iter]
