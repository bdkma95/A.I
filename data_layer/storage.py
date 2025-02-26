import psycopg2
import pandas as pd
import boto3
import os
import pyarrow.parquet as pq
import ssl
from typing import Optional, Dict, Any, List, Union
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import execute_batch
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.logger import logger
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
import io

class DatabaseManager:
    """Base class for database operations with common interface"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = self._configure_encryption(config)
        self.engine = self.create_engine()
        
    def _configure_encryption(self, config: Dict) -> Dict:
        """Configure encryption parameters for database connections"""
        config.setdefault('sslmode', 'require')
        config.setdefault('sslrootcert', os.getenv('DB_SSL_CA'))
        return config
        
    def create_engine(self) -> Engine:
        """Create SQLAlchemy engine for database backend"""
        raise NotImplementedError
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute read query and return results as DataFrame"""
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)
        
    def bulk_delete(self, table: str, where_clause: str, params: List) -> int:
        """Perform bulk delete operation with validation"""
        if not where_clause:
            raise ValueError("Where clause required for bulk delete")
            
        query = f"DELETE FROM {table} WHERE {where_clause}"
        return self.execute_update(query, params)
        
    def execute_update(self, query: str, params: List = None) -> int:
        """Execute update/delete query and return affected rows"""
        with self.engine.begin() as conn:
            result = conn.execute(text(query), params or {})
            return result.rowcount

class PostgresManager(DatabaseManager):
    """PostgreSQL implementation with connection pooling and health checks"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pool = self._create_connection_pool()
        self.connection_ages = {}
        
    def create_engine(self) -> Engine:
        return create_engine(f"postgresql+psycopg2://{self.config['user']}:{self.config['password']}"
                            f"@{self.config['host']}:{self.config['port']}/{self.config['database']}",
                            pool_size=10, max_overflow=20)

    def _create_connection_pool(self) -> SimpleConnectionPool:
        """Create connection pool with health checks"""
        pool = SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            **self.config
        )
        self._test_pool_health(pool)
        return pool
        
    def _test_pool_health(self, pool: SimpleConnectionPool) -> bool:
        """Verify all connections in the pool are healthy"""
        for _ in range(pool.minconn):
            conn = pool.getconn()
            if not self._is_connection_alive(conn):
                conn.close()
                pool.putconn(conn)
        return True
        
    def _is_connection_alive(self, conn) -> bool:
        """Check if connection is still valid"""
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone()[0] == 1
        except psycopg2.InterfaceError:
            return False
            
    def _recycle_connections(self) -> None:
        """Recycle connections older than max age"""
        max_age = timedelta(minutes=30)
        now = datetime.now()
        
        for conn in list(self.connection_ages.keys()):
            if now - self.connection_ages[conn] > max_age:
                conn.close()
                del self.connection_ages[conn]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def save_data(self, data: pd.DataFrame, table: str, batch_size: int = 1000) -> None:
        """Save DataFrame to PostgreSQL with connection recycling"""
        self._recycle_connections()
        columns = data.columns.tolist()
        query = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join(['%s']*len(columns))})"
        
        with self.engine.begin() as conn:
            data.to_sql(table, conn, if_exists='append', index=False, method='multi')

class S3Manager:
    """S3 Manager with enhanced data handling and validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.client = boto3.client('s3', **config)
        self.config = config
        
    def upload_dataframe(
        self,
        df: pd.DataFrame,
        bucket: str,
        key: str,
        format: str = 'parquet',
        validation_schema: Dict = None
    ) -> str:
        """Upload DataFrame with format conversion and validation"""
        self._validate_data(df, validation_schema)
        
        buffer = io.BytesIO()
        if format == 'parquet':
            df.to_parquet(buffer, engine='pyarrow')
        elif format == 'csv':
            df.to_csv(buffer, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        buffer.seek(0)
        self.client.put_object(
            Bucket=bucket,
            Key=key,
            Body=buffer,
            ServerSideEncryption='AES256'
        )
        return f"s3://{bucket}/{key}"
        
    def _validate_data(self, df: pd.DataFrame, schema: Dict) -> None:
        """Perform data validation against schema"""
        if schema:
            missing = [col for col in schema['required'] if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
                
            type_mismatch = []
            for col, dtype in schema['dtypes'].items():
                if df[col].dtype != dtype:
                    type_mismatch.append(col)
            if type_mismatch:
                raise ValueError(f"Type mismatch in columns: {type_mismatch}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def upload_file(
        self,
        local_path: str,
        bucket: str,
        key: str,
        convert_to: str = None
    ) -> str:
        """Upload file with optional format conversion"""
        if convert_to:
            if not local_path.endswith('.csv'):
                raise ValueError("Conversion only supported from CSV")
                
            df = pd.read_csv(local_path)
            return self.upload_dataframe(df, bucket, key, format=convert_to)
            
        self.client.upload_file(
            Filename=local_path,
            Bucket=bucket,
            Key=key,
            ExtraArgs={'ServerSideEncryption': 'AES256'}
        )
        return f"s3://{bucket}/{key}"

class MySQLManager(DatabaseManager):
    """MySQL database implementation"""
    
    def create_engine(self) -> Engine:
        return create_engine(f"mysql+pymysql://{self.config['user']}:{self.config['password']}"
                            f"@{self.config['host']}:{self.config['port']}/{self.config['database']}",
                            pool_size=10, pool_recycle=3600)

class SQLiteManager(DatabaseManager):
    """SQLite database implementation"""
    
    def create_engine(self) -> Engine:
        return create_engine(f"sqlite:///{self.config['database']}", pool_size=10)

# Usage Example
if __name__ == "__main__":
    # PostgreSQL with encryption
    pg_config = {
        "host": "db-host",
        "database": "football",
        "user": "user",
        "password": "password",
        "port": 5432,
        "sslmode": "verify-full",
        "sslrootcert": "/path/to/ca-cert"
    }
    pg_manager = PostgresManager(pg_config)
    
    # Execute complex query
    results = pg_manager.execute_query("""
        SELECT player_id, AVG(goals) 
        FROM matches 
        WHERE season = :season
        GROUP BY player_id
    """, {'season': 2023})
    
    # Bulk delete old records
    pg_manager.bulk_delete("matches", "match_date < %s", [datetime(2022, 1, 1)])
    
    # S3 Parquet upload with validation
    s3_manager = S3Manager({
        "aws_access_key_id": "key",
        "aws_secret_access_key": "secret",
        "region_name": "us-west-1"
    })
    
    schema = {
        "required": ["player_id", "goals"],
        "dtypes": {"player_id": "int64", "goals": "float64"}
    }
    s3_manager.upload_dataframe(
        df=results,
        bucket="analytics",
        key="player_stats.parquet",
        validation_schema=schema
    )
