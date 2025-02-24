import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import execute_batch
import boto3

class PostgresManager:
    def __init__(self, config):
        self.pool = SimpleConnectionPool(1, 10, **config)
        
    def save_data(self, data: pd.DataFrame, table_name: str):
        conn = self.pool.getconn()
        """Save data to PostgreSQL using connection pooling."""
        conn = postgres_pool.getconn()
        cur = conn.cursor()
        try:
            columns = data.columns.tolist()
            template = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({','.join(['%s']*len(columns))})"
            execute_batch(cur, template, data.values.tolist())
            conn.commit()
        except Exception as e:
            logger.error(f"Database Error: {str(e)}")
            conn.rollback()
        finally:
            cur.close()
            postgres_pool.putconn(conn)

class S3Manager:
    def __init__(self, config):
        self.client = boto3.client('s3', **config)
        
    def upload_file(self, file_path: str, bucket: str):
        None
        """Upload a file to AWS S3."""
        if object_name is None:
            object_name = os.path.basename(file_name)
        s3 = boto3.client('s3', **self.aws_config)
        try:
            s3.upload_file(file_name, bucket_name, object_name)
            logger.info(f"File {file_name} uploaded to S3 bucket {bucket_name} as {object_name}.")
        except Exception as e:
            logger.error(f"S3 Upload Error: {str(e)}")
