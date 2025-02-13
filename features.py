from datetime import timedelta
from feast import Entity, FeatureView, Field, ValueType
from feast.types import Float32, Int64
from feast.infra.offline_stores.file_source import FileSource

# Define entities
player = Entity(name="player_id", value_type=ValueType.INT64)

# Define data sources
player_stats_source = FileSource(
    path="data/player_stats.parquet",
    event_timestamp_column="event_timestamp",
)

# Define feature views
player_stats_fv = FeatureView(
    name="player_stats",
    entities=[player],
    ttl=timedelta(days=365),
    schema=[
        Field(name="goals", dtype=Int64),
        Field(name="assists", dtype=Int64),
        Field(name="xg_contribution", dtype=Float32),
    ],
    source=player_stats_source,
)
