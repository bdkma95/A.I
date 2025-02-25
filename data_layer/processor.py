import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from pydantic import BaseModel, ValidationError
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from feast import FeatureStore
import great_expectations as ge
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------
# Schema Validation
# ---------------------
class FootballDataSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    expected_goals: pd.Series
    expected_assists: pd.Series
    successful_passes: pd.Series
    total_passes: pd.Series
    match_date: pd.Series
    
    @classmethod
    def validate_df(cls, df: pd.DataFrame) -> bool:
        try:
            cls(**df)
            return True
        except ValidationError as e:
            logger.error(f"Schema validation failed: {str(e)}")
            raise

# ---------------------
# Data Processor Class
# ---------------------
class DataProcessor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'feature_selection': {
                'method': 'mutual_info',
                'top_k': 10
            },
            'quality_metrics': {
                'completeness_threshold': 0.95,
                'value_ranges': {
                    'pass_accuracy': (0.0, 1.0)
                }
            }
        }
        self.feature_store = FeatureStore(repo_path="feature_repo/")
        self.quality_report: Dict = {}
        self.feature_importances: pd.DataFrame = None

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Main data processing pipeline"""
        self._validate_input(data)
        processed_data = self._core_processing(data.copy())
        self._post_processing(processed_data)
        return processed_data

    def _validate_input(self, data: pd.DataFrame) -> None:
        """Validate input data structure and schema"""
        FootballDataSchema.validate_df(data)
        self._check_feature_store_schema(data)

    def _core_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute core processing steps"""
        data = self._handle_missing_values(data)
        data = self._create_features(data)
        data = self._select_features(data)
        self._calculate_feature_importance(data)
        return data

    def _post_processing(self, data: pd.DataFrame) -> None:
        """Execute post-processing tasks"""
        self._generate_quality_report(data)
        self._save_to_feature_store(data)
        
    # ---------------------
    # Feature Engineering
    # ---------------------
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create derived features"""
        data['xg_contribution'] = data['expected_goals'] + data['expected_assists']
        data['pass_accuracy'] = data['successful_passes'] / data['total_passes']
        data = self._create_temporal_features(data)
        return data

    def _create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        data['match_week'] = data['match_date'].dt.isocalendar().week
        data['month'] = data['match_date'].dt.month
        return data

    # ---------------------
    # Feature Selection
    # ---------------------
    def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select top features based on config"""
        method = self.config['feature_selection']['method']
        k = self.config['feature_selection']['top_k']
        
        if method == 'mutual_info':
            return self._select_features_mutual_info(data, k)
        elif method == 'importance':
            return self._select_features_by_importance(data, k)
        else:
            logger.warning("No valid feature selection method specified")
            return data

    def _select_features_mutual_info(self, data: pd.DataFrame, k: int) -> pd.DataFrame:
        """Select features using mutual information"""
        selector = SelectKBest(mutual_info_regression, k=k)
        features = selector.fit_transform(data.drop('target', axis=1), data['target'])
        selected_cols = data.columns[selector.get_support()]
        return data[selected_cols]

    # ---------------------
    # Feature Importance
    # ---------------------
    def _calculate_feature_importance(self, data: pd.DataFrame) -> None:
        """Calculate and store feature importances"""
        model = RandomForestRegressor()
        model.fit(data.drop('target', axis=1), data['target'])
        self.feature_importances = pd.DataFrame({
            'feature': data.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get calculated feature importances"""
        if self.feature_importances is None:
            raise ValueError("Feature importance not calculated. Run process_data first")
        return self.feature_importances

    # ---------------------
    # Feature Store Integration
    # ---------------------
    def _save_to_feature_store(self, data: pd.DataFrame) -> None:
        """Save processed features to Feast feature store"""
        self.feature_store.write_features(
            entity_rows=data.to_dict('records'),
            features=[
                "player_stats:xg_contribution",
                "player_stats:pass_accuracy"
            ]
        )

    def _check_feature_store_schema(self, data: pd.DataFrame) -> None:
        """Validate against feature store schema"""
        fs_schema = self.feature_store.get_feature_service("player_stats").features
        missing_features = [f.name for f in fs_schema if f.name not in data.columns]
        if missing_features:
            logger.warning(f"Missing features required by feature store: {missing_features}")

    # ---------------------
    # Data Quality
    # ---------------------
    def _generate_quality_report(self, data: pd.DataFrame) -> None:
        """Generate comprehensive data quality report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'completeness': self._calculate_completeness(data),
                'value_ranges': self._check_value_ranges(data),
                'outliers': self._detect_outliers(data)
            },
            'schema_validation': self._validate_output_schema(data)
        }
        self.quality_report = report
        self._save_quality_report(report)

    def _calculate_completeness(self, data: pd.DataFrame) -> Dict:
        """Calculate completeness metrics"""
        return data.notnull().mean().to_dict()

    def _check_value_ranges(self, data: pd.DataFrame) -> Dict:
        """Validate value ranges against config"""
        results = {}
        for col, (min_val, max_val) in self.config['quality_metrics']['value_ranges'].items():
            if col in data.columns:
                results[col] = {
                    'pass_rate': data[col].between(min_val, max_val).mean(),
                    'violations': data[~data[col].between(min_val, max_val)][col].count()
                }
        return results

    def _save_quality_report(self, report: Dict) -> None:
        """Persist quality report"""
        with open(f"quality_reports/report_{datetime.now().date()}.json", "w") as f:
            json.dump(report, f)

    # ---------------------
    # Type Hinting
    # ---------------------
    def get_quality_report(self) -> Dict[str, Union[Dict, str]]:
        """Get generated quality report"""
        return self.quality_report

    def get_processed_schema(self) -> Dict[str, str]:
        """Get schema of processed data"""
        return {
            'xg_contribution': 'float64',
            'pass_accuracy': 'float64',
            'match_week': 'int64',
            'month': 'int64'
        }
