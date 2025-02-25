import pandas as pd
import numpy as np
import mlflow
import shap
from typing import Optional, Dict, List, Tuple, Union
from pydantic import BaseModel, Field
from datetime import datetime
import hashlib
import json
import pytest
from packaging.version import Version

# ---------------------
# Versioning & Lineage
# ---------------------
class FeatureVersion(BaseModel):
    version: str = Field(..., regex=r'^\d+\.\d+\.\d+$')
    schema: Dict[str, str]
    description: str
    created_at: datetime = datetime.now()
    
class DataLineage(BaseModel):
    input_hash: str
    processing_steps: List[str]
    feature_versions: Dict[str, FeatureVersion]
    parameters: Dict
    parent_lineage: Optional[str] = None

# ---------------------
# Core Processor Class
# ---------------------
class FootballDataProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.lineage: DataLineage = None
        self.current_version = Version("1.2.0")
        self._init_version_history()
        
    def _init_version_history(self):
        self.version_history = {
            "1.0.0": FeatureVersion(
                version="1.0.0",
                schema={"xg_contribution": "float", "pass_accuracy": "float"},
                description="Initial feature set"
            ),
            "1.1.0": FeatureVersion(
                version="1.1.0",
                schema={"xg_contribution": "float", "pass_accuracy": "float", "defensive_impact": "float"},
                description="Added defensive impact metric"
            ),
            "1.2.0": FeatureVersion(
                version="1.2.0",
                schema={**self.version_history["1.1.0"].schema, "physical_load": "float"},
                description="Added physical load metric from wearable data"
            )
        }

    def process_data(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Main processing pipeline with versioning and tracking"""
        with mlflow.start_run():
            try:
                # Track lineage and version
                self._create_lineage(raw_data)
                
                # Schema evolution handling
                raw_data = self._handle_schema_evolution(raw_data)
                
                # Core processing
                processed_data = self._apply_processing_steps(raw_data)
                
                # Explainability
                explain_report = self._generate_explainability_report(processed_data)
                
                # MLflow tracking
                self._log_mlflow_artifacts(processed_data, explain_report)
                
                return processed_data, explain_report
                
            except Exception as e:
                mlflow.log_param("error", str(e))
                raise

    def _create_lineage(self, data: pd.DataFrame) -> None:
        """Create data lineage record"""
        input_hash = hashlib.sha256(pd.util.hash_pandas_object(data).hexdigest())
        self.lineage = DataLineage(
            input_hash=input_hash,
            processing_steps=[],
            feature_versions={},
            parameters=self.config
        )

    def _handle_schema_evolution(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply schema migrations if needed"""
        if 'wearable_id' in data.columns and self.current_version >= Version("1.2.0"):
            data = self._add_physical_load(data)
        return data

    def _apply_processing_steps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute versioned processing steps"""
        # Base features
        data = self._create_base_features(data)
        
        # Version-specific features
        if self.current_version >= Version("1.1.0"):
            data = self._create_defensive_metrics(data)
            
        if self.current_version >= Version("1.2.0"):
            data = self._create_physical_metrics(data)
            
        return data

    def _generate_explainability_report(self, data: pd.DataFrame) -> Dict:
        """Generate SHAP explainability report"""
        explainer = shap.Explainer(data.drop('target', axis=1))
        shap_values = explainer(data)
        
        return {
            "shap_summary": shap_values.summary_plot(),
            "feature_importance": self._calculate_feature_importance(data),
            "version": str(self.current_version)
        }

    def _log_mlflow_artifacts(self, data: pd.DataFrame, report: Dict) -> None:
        """Log processing artifacts to MLflow"""
        mlflow.log_params(self.config)
        mlflow.log_metrics(report['feature_importance']['top_features'])
        
        # Log datasets
        data.to_parquet("processed_data.parquet")
        mlflow.log_artifact("processed_data.parquet")
        
        # Log explainability
        with open("shap_report.html", "w") as f:
            f.write(report['shap_summary'])
        mlflow.log_artifact("shap_report.html")

    # ---------------------
    # Feature Engineering
    # ---------------------
    def _create_base_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Version 1.0 features"""
        data['xg_contribution'] = data['expected_goals'] + data['expected_assists']
        data['pass_accuracy'] = data['successful_passes'] / data['total_passes']
        self.lineage.processing_steps.append("base_features_v1")
        return data

    def _create_defensive_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Version 1.1 features"""
        data['defensive_impact'] = 0.7*data['tackles'] + 0.3*data['interceptions']
        self.lineage.processing_steps.append("defensive_metrics_v1.1")
        return data

    def _create_physical_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Version 1.2 features"""
        data['physical_load'] = data['distance_covered'] * data['sprint_intensity']
        self.lineage.processing_steps.append("physical_metrics_v1.2")
        return data

# ---------------------
# Unit Tests
# ---------------------
class TestFootballDataProcessor:
    @pytest.fixture
    def processor(self):
        return FootballDataProcessor(config={"version": "1.2.0"})

    @pytest.fixture
    def legacy_data(self):
        return pd.DataFrame({
            'expected_goals': [1.2, 0.8],
            'expected_assists': [0.4, 0.3],
            'successful_passes': [45, 32],
            'total_passes': [50, 35]
        })

    def test_backward_compatibility(self, processor, legacy_data):
        processed, _ = processor.process_data(legacy_data)
        assert 'xg_contribution' in processed.columns
        assert 'defensive_impact' not in processed.columns

    def test_schema_evolution(self, processor):
        new_data = pd.DataFrame({
            'expected_goals': [1.5],
            'expected_assists': [0.6],
            'successful_passes': [50],
            'total_passes': [55],
            'wearable_id': [123],
            'distance_covered': [10500],
            'sprint_intensity': [0.85]
        })
        processed, _ = processor.process_data(new_data)
        assert 'physical_load' in processed.columns

    def test_missing_data_handling(self, processor):
        incomplete_data = pd.DataFrame({
            'expected_goals': [None, 0.8],
            'expected_assists': [0.4, None]
        })
        with pytest.raises(DataValidationError):
            processor.process_data(incomplete_data)

    def test_extreme_values(self, processor):
        edge_data = pd.DataFrame({
            'expected_goals': [1e6, -1],
            'expected_assists': [None, 5000]
        })
        with pytest.raises(DataValidationError):
            processor.process_data(edge_data)

# ---------------------
# Schema Evolution Strategies
# ---------------------
class SchemaManager:
    def __init__(self):
        self.migration_scripts = {
            ("1.0.0", "1.1.0"): self._migrate_v1_to_v1_1,
            ("1.1.0", "1.2.0"): self._migrate_v1_1_to_v1_2
        }

    def migrate(self, data: pd.DataFrame, from_version: str, to_version: str) -> pd.DataFrame:
        current_version = Version(from_version)
        target_version = Version(to_version)
        
        while current_version < target_version:
            next_version = self._get_next_version(current_version)
            migration_fn = self.migration_scripts.get((str(current_version), str(next_version)))
            if migration_fn:
                data = migration_fn(data)
            current_version = next_version
            
        return data

    def _migrate_v1_to_v1_1(self, data: pd.DataFrame) -> pd.DataFrame:
        # Add defensive impact default value
        if 'defensive_impact' not in data.columns:
            data['defensive_impact'] = 0.0
        return data

    def _migrate_v1_1_to_v1_2(self, data: pd.DataFrame) -> pd.DataFrame:
        # Calculate physical load if possible
        if {'distance_covered', 'sprint_intensity'}.issubset(data.columns):
            data['physical_load'] = data['distance_covered'] * data['sprint_intensity']
        return data
