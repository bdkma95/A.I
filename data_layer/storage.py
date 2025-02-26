import psycopg2
import pandas as pd
import boto3
import os
from typing import Optional, Dict, Any
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import execute_batch
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.logger import logger
from typing import List

class PostgresManager:
    """Managed PostgreSQL connection pool with retry logic and batch operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool = self._create_connection_pool()
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _create_connection_pool(self) -> SimpleConnectionPool:
        """Create a connection pool with retry logic"""
        try:
            return SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                **self.config
            )
        except psycopg2.OperationalError as e:
            logger.error(f"Connection pool creation failed: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def save_data(self, data: pd.DataFrame, table_name: str, batch_size: int = 1000) -> None:
        """Save DataFrame to PostgreSQL with transaction management"""
        columns = data.columns.tolist()
        template = self._prepare_insert_template(table_name, columns)
        records = data.to_records(index=False).tolist()

        with self.pool.getconn() as conn:
            try:
                with conn.cursor() as cur:
                    execute_batch(cur, template, records, page_size=batch_size)
                    conn.commit()
                    logger.info(f"Inserted {len(data)} rows into {table_name}")
            except psycopg2.Error as e:
                logger.error(f"Database error: {str(e)}")
                conn.rollback()
                raise
            finally:
                self.pool.putconn(conn)

    def _prepare_insert_template(self, table_name: str, columns: List[str]) -> str:
        """Generate SQL insert template with parameter placeholders"""
        columns_str = ','.join(columns)
        placeholders = ','.join(['%s'] * len(columns))
        return f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"

    def close_pool(self) -> None:
        """Close all connections in the pool"""
        self.pool.closeall()

class S3Manager:
    """Managed S3 operations with validation and error handling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.client = boto3.client('s3', **config)
        self.config = config

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def upload_file(
        self,
        file_path: str,
        bucket_name: str,
        object_name: Optional[str] = None,
        extra_args: Optional[Dict] = None
    ) -> str:
        """Upload a file to S3 with validation and metadata support"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        object_name = object_name or os.path.basename(file_path)
        
        try:
            self.client.upload_file(
                Filename=file_path,
                Bucket=bucket_name,
                Key=object_name,
                ExtraArgs=extra_args or {}
            )
            s3_url = f"s3://{bucket_name}/{object_name}"
            logger.info(f"Successfully uploaded to {s3_url}")
            return s3_url
        except Exception as e:
            logger.error(f"S3 upload failed: {str(e)}")
            raise

    def download_file(self, bucket_name: str, object_name: str, file_path: str) -> None:
        """Download a file from S3 with validation"""
        try:
            self.client.download_file(
                Bucket=bucket_name,
                Key=object_name,
                Filename=file_path
            )
            logger.info(f"Downloaded {object_name} to {file_path}")
        except Exception as e:
            logger.error(f"S3 download failed: {str(e)}")
            raise

    def generate_presigned_url(self, bucket_name: str, object_name: str, expiration: int = 3600) -> str:
        """Generate a presigned URL for secure access"""
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': object_name},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error(f"Presigned URL generation failed: {str(e)}")
            raise
