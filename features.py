from datetime import timedelta, datetime
from feast import Entity, FeatureView, Field, ValueType, FeatureService
from feast.types import Float32, Int64, Bool, String
from feast.infra.offline_stores.file_source import FileSource
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource
from feast.data_format import ParquetFormat
from typing import Optional
import pandas as pd

# ======================
# Enhanced Entities
# ======================
player = Entity(
    name="player_id",
    description="Unique identifier for a player",
    value_type=ValueType.INT64,
    tags={"domain": "player_performance"},
)

match = Entity(
    name="match_id",
    description="Unique identifier for a match",
    value_type=ValueType.INT64,
    tags={"domain": "match_data"},
)

# ======================
# Composite Key Entity
# ======================
player_match = Entity(
    name="player_match",
    description="Composite key combining player and match",
    value_type=ValueType.STRING,
    tags={"composite_key": "true"},
    join_keys=["player_id", "match_id"],
)

# ======================
# Data Sources
# ======================
player_stats_source = PostgreSQLSource(
    name="player_stats_db",
    query="""
    SELECT
        player_id,
        match_id,
        event_timestamp,
        created_timestamp,
        goals,
        assists,
        xg_contribution,
        pass_accuracy,
        defensive_score,
        injury_risk
    FROM player_performance
    """,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

player_biometrics_source = FileSource(
    name="player_biometrics",
    path="gs://football-data/player_biometrics.parquet",
    file_format=ParquetFormat(),
    timestamp_field="measurement_time",
    description="Player biometric data from wearable devices",
)

# ======================
# Feature Views
# ======================
player_performance_fv = FeatureView(
    name="player_performance",
    entities=[player_match],
    ttl=timedelta(days=180),
    schema=[
        Field(name="goals", dtype=Int64),
        Field(name="assists", dtype=Int64),
        Field(name="xg_contribution", dtype=Float32),
        Field(name="pass_accuracy", dtype=Float32),
        Field(name="defensive_score", dtype=Float32),
        Field(name="injury_risk", dtype=Float32),
        Field(name="is_starter", dtype=Bool),
    ],
    source=player_stats_source,
    tags={"team": "analytics"},
)

player_biometrics_fv = FeatureView(
    name="player_biometrics",
    entities=[player],
    ttl=timedelta(days=7),  # Shorter TTL for time-sensitive biometric data
    schema=[
        Field(name="heart_rate", dtype=Int64),
        Field(name="distance_covered", dtype=Float32),
        Field(name="sprint_speed", dtype=Float32),
        Field(name="fatigue_level", dtype=Float32),
    ],
    source=player_biometrics_source,
    online=True,  # Enable real-time access
    tags={"source": "wearables"},
)

# ======================
# Feature Services
# ======================
player_analytics_service = FeatureService(
    name="player_analytics",
    features=[
        player_performance_fv,
        player_biometrics_fv[["distance_covered", "sprint_speed"]],
    ],
    tags={"purpose": "model_serving"},
)

injury_prediction_service = FeatureService(
    name="injury_prediction",
    features=[
        player_biometrics_fv,
        player_performance_fv[["injury_risk", "defensive_score"]],
    ],
    tags={"purpose": "health_monitoring"},
)

# ======================
# Validation & Utilities
# ======================
def validate_feature_view(fv: FeatureView):
    """Validate feature view schema and metadata"""
    required_tags = ["team", "source"]
    for tag in required_tags:
        if tag not in fv.tags:
            raise ValueError(f"Missing required tag '{tag}' in {fv.name}")
    
    if fv.ttl > timedelta(days=365):
        raise ValueError("TTL cannot exceed 1 year for compliance reasons")

def generate_feature_documentation(feature_service: FeatureService) -> str:
    """Generate markdown documentation for feature services"""
    docs = f"# {feature_service.name}\n\n"
    docs += f"**Description**: {feature_service.description}\n\n"
    docs += "## Included Features:\n"
    
    for feature in feature_service.features:
        docs += f"- {feature.name} ({feature.dtype})\n"
    
    return docs

# ======================
# Temporal Features
# ======================
def create_rolling_window_features(
    df: pd.DataFrame,
    window_sizes: list = [3, 5, 10],
    metrics: list = ["goals", "assists"]
) -> pd.DataFrame:
    """Generate rolling window features for time-series analysis"""
    for window in window_sizes:
        for metric in metrics:
            df[f"{metric}_rolling_{window}"] = (
                df.groupby("player_id")[metric]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
    return df

# ======================
# Unit Tests (Pytest-style)
# ======================
def test_feature_service_completeness():
    assert len(player_analytics_service.features) >= 5
    assert "distance_covered" in [f.name for f in player_analytics_service.features]
