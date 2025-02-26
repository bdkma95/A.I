import psycopg2
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import boto3
import os
import time
import zlib
from typing import Dict, List, Optional, Any, Union
from prometheus_client import Gauge, Histogram, start_http_server
from cryptography.fernet import Fernet
from alembic import config as alembic_config, command
from sqlalchemy import create_engine, event, text
from threading import Lock
from queue import Queue
import json

# ======================
# Metrics & Monitoring
# ======================
DB_CONNECTIONS = Gauge('db_connections', 'Current database connections', ['state'])
QUERY_DURATION = Histogram('query_duration', 'Query execution time', ['operation'])
POOL_WAIT_TIME = Histogram('pool_wait_time', 'Connection wait time')

# ======================
# Columnar Encryption
# ======================
class ColumnarEncryption:
    def __init__(self, key_store: Dict[str, bytes]):
        self.keys = key_store
        self.column_crypto = {}

    def configure_encryption(self, column_map: Dict[str, str]):
        """Map columns to encryption keys"""
        self.column_crypto = {
            col: self.keys[key_name] for col, key_name in column_map.items()
        }

    def get_encryption_config(self):
        """Generate parquet encryption configuration"""
        return pq.EncryptionConfiguration(
            column_keys=self.column_crypto,
            encryption_algorithm="AES_GCM_V1",
            cache_lifetime=300
        )

# ======================
# Database Manager Base
# ======================
class DatabaseManager:
    def __init__(self, config: Dict):
        self.config = config
        self.migration_lock = Lock()
        self.load_balancer = ConnectionLoadBalancer(config)
        self._init_metrics()

    def _init_metrics(self):
        start_http_server(9000)
        event.listen(self.engine, 'connect', self._track_connection)
        event.listen(self.engine, 'close', self._track_disconnect)

    def _track_connection(self, conn, branch):
        DB_CONNECTIONS.labels(state='active').inc()

    def _track_disconnect(self, conn):
        DB_CONNECTIONS.labels(state='active').dec()

    @QUERY_DURATION.labels(operation='migration').time()
    def run_migrations(self):
        """Execute schema migrations using Alembic"""
        with self.migration_lock:
            alembic_cfg = alembic_config.Config("alembic.ini")
            alembic_cfg.attributes['configure_logger'] = False
            command.upgrade(alembic_cfg, "head")

    def track_changes(self, table: str):
        """Enable change data capture for a table"""
        self.execute(f"""
            ALTER TABLE {table} 
            REPLICA IDENTITY FULL;
            CREATE PUBLICATION {table}_pub 
            FOR TABLE {table} WITH (publish = 'insert,update,delete');
        """)

    def get_changes(self, slot_name: str) -> List[Dict]:
        """Retrieve captured changes"""
        return self.load_balancer.execute_query(
            "SELECT * FROM pg_logical_slot_get_changes(%s, NULL, NULL);",
            (slot_name,)
        )

# ======================
# Connection Load Balancing
# ======================
class ConnectionLoadBalancer:
    def __init__(self, config: Dict):
        self.read_replicas = config.get('read_replicas', [])
        self.write_engine = create_engine(config['primary'])
        self.read_engines = [create_engine(replica) for replica in self.read_replicas]
        self.replica_index = 0
        self.lock = Lock()

    def get_read_connection(self):
        """Round-robin read replica selection"""
        with self.lock:
            engine = self.read_engines[self.replica_index]
            self.replica_index = (self.replica_index + 1) % len(self.read_engines)
            return engine.connect()

    def execute_query(self, query: str, params=None, read_only=True):
        """Route query to appropriate connection"""
        if read_only and self.read_replicas:
            with self.get_read_connection() as conn:
                return conn.execute(text(query), params or {})
        else:
            with self.write_engine.connect() as conn:
                return conn.execute(text(query), params or {})

# ======================
# S3 Manager Enhancements
# ======================
class S3Manager:
    COMPRESSION_OPTIONS = {
        'snappy': {'compression': 'snappy'},
        'gzip': {'compression': 'gzip'},
        'brotli': {'compression': 'brotli'}
    }

    def __init__(self, config: Dict):
        self.client = boto3.client('s3', **config)
        self.encryption = ColumnarEncryption(config.get('encryption_keys', {}))

    @QUERY_DURATION.labels(operation='s3_upload').time()
    def upload_dataframe(
        self,
        df: pd.DataFrame,
        bucket: str,
        key: str,
        compression: str = 'snappy',
        encryption_map: Dict[str, str] = None
    ) -> str:
        """Upload DataFrame with compression and encryption"""
        table = pa.Table.from_pandas(df)
        buf = pa.BufferOutputStream()
        
        crypto_config = None
        if encryption_map:
            self.encryption.configure_encryption(encryption_map)
            crypto_config = self.encryption.get_encryption_config()

        pq.write_table(
            table,
            buf,
            compression=self.COMPRESSION_OPTIONS.get(compression, 'snappy'),
            encryption_config=crypto_config
        )
        
        self.client.put_object(
            Bucket=bucket,
            Key=key,
            Body=buf.getvalue().to_pybytes(),
            ServerSideEncryption='AES256'
        )
        return f"s3://{bucket}/{key}"

# ======================
# Change Data Capture
# ======================
class ChangeCapture:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.change_queue = Queue(maxsize=1000)
        self.running = False

    def start_capture(self, slot_name: str):
        """Start streaming changes from logical replication slot"""
        self.running = True
        while self.running:
            changes = self.db.get_changes(slot_name)
            for change in changes:
                self.change_queue.put(self._parse_change(change))
            time.sleep(1)

    def _parse_change(self, change: Dict) -> Dict:
        """Parse WAL change data"""
        return {
            'operation': change['action'],
            'timestamp': change['stamp'],
            'data': json.loads(change['data'])
        }

    def get_changes(self) -> List[Dict]:
        """Retrieve captured changes"""
        return list(self.change_queue.queue)
