from datetime import timedelta, datetime
from feast import Entity, FeatureView, Field, ValueType, FeatureService, KafkaSource, KinesisSource
from feast.types import Float32, Int64, Bool, String
from feast.infra.offline_stores.file_source import FileSource
from feast.data_format import ParquetFormat, AvroFormat
from feast.value_type import ValueType
from cryptography.fernet import Fernet
from typing import Optional, Dict, List
import pandas as pd
import great_expectations as ge
from pyspark.sql import SparkSession
import json

# ======================
# Encryption Setup
# ======================
encryption_key = Fernet.generate_key()
cipher_suite = Fernet(encryption_key)

class EncryptedField(Field):
    def __init__(self, name: str, dtype: ValueType):
        super().__init__(name, dtype)
        
    def encrypt(self, value: str) -> bytes:
        return cipher_suite.encrypt(value.encode())
    
    def decrypt(self, token: bytes) -> str:
        return cipher_suite.decrypt(token).decode()

# ======================
# Versioned Feature Views
# ======================
class VersionedFeatureView(FeatureView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version = kwargs.get('version', '1.0.0')
        self.tags = self.tags or {}
        self.tags.update({
            'version': self.version,
            'deprecated': 'false',
            'compatibility': 'backward'
        })

# ======================
# Streaming Sources
# ======================
player_biometrics_stream = KafkaSource(
    name="player_biometrics_stream",
    kafka_bootstrap_servers="kafka:9092",
    topic="player_biometrics",
    message_format=AvroFormat(
        schema_json=json.dumps({
            "type": "record",
            "name": "BiometricRecord",
            "fields": [
                {"name": "player_id", "type": "int"},
                {"name": "heart_rate", "type": "int"},
                {"name": "timestamp", "type": "long"}
            ]
        })
    ),
    timestamp_field="timestamp",
    watermark_delay_threshold=timedelta(minutes=5),
    description="Real-time biometric data from wearables",
)

# ======================
# Feature Monitoring
# ======================
class DataQualityCheck:
    def __init__(self, feature_view: FeatureView):
        self.feature_view = feature_view
        self.expectation_suite = ge.core.ExpectationSuite(
            f"{feature_view.name}_quality_checks"
        )
        
    def add_range_check(self, field: str, min_val: float, max_val: float):
        self.expectation_suite.add_expectation(
            ge.core.ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": field,
                    "min_value": min_val,
                    "max_value": max_val
                }
            )
        )
    
    def run_checks(self, df: pd.DataFrame) -> bool:
        report = ge.from_pandas(df).validate(self.expectation_suite)
        return report.success

def with_data_quality_checks(func):
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        fv = args[0]
        checker = DataQualityCheck(fv)
        
        # Add common checks
        if 'heart_rate' in df.columns:
            checker.add_range_check('heart_rate', 40, 220)
        if 'distance_covered' in df.columns:
            checker.add_range_check('distance_covered', 0, 15000)
            
        if not checker.run_checks(df):
            raise ValueError("Data quality checks failed")
        return df
    return wrapper

# ======================
# Enhanced Feature Views
# ======================
class SecureFeatureView(VersionedFeatureView):
    @with_data_quality_checks
    def materialize(self, start_date: datetime, end_date: datetime):
        # Implementation with quality checks
        pass

player_biometrics_fv = SecureFeatureView(
    name="player_biometrics_v2",
    version="2.1.0",
    entities=[player],
    ttl=timedelta(days=7),
    schema=[
        EncryptedField(name="heart_rate", dtype=Int64),
        Field(name="distance_covered", dtype=Float32),
        EncryptedField(name="sleep_quality", dtype=Float32),
    ],
    source=player_biometrics_stream,
    online=True,
    tags={
        "sensitivity": "high",
        "encryption": "fernet-256",
        "retention_policy": "30d"
    },
)

# ======================
# Validation Pipeline
# ======================
class FeatureValidationPipeline:
    def __init__(self):
        self.checks = []
        self.metrics = {}
        
    def add_check(self, check_fn):
        self.checks.append(check_fn)
        
    def run_validation(self, df: pd.DataFrame):
        results = {}
        for check in self.checks:
            results[check.__name__] = check(df)
        self.metrics = self._calculate_metrics(df)
        return all(results.values())
    
    def _calculate_metrics(self, df):
        return {
            'row_count': len(df),
            'null_percentages': df.isnull().mean().to_dict(),
            'mean_values': df.mean().to_dict(),
            'std_dev': df.std().to_dict()
        }

# ======================
# Streaming Integration
# ======================
def create_spark_streaming_pipeline():
    spark = SparkSession.builder \
        .appName("BiometricStreamProcessor") \
        .config("spark.sql.streaming.checkpointLocation", "/checkpoints") \
        .getOrCreate()

    return spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:9092") \
        .option("subscribe", "player_biometrics") \
        .load()
