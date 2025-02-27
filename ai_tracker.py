import signal
import sys
from typing import Optional, Dict, Any
from flask import Flask
from flask_socketio import SocketIO
from flask_limiter import Limiter
from flasgger import Swagger
from prometheus_client import start_http_server
from config.settings import settings
from data.collector import DataCollector
from data.processor import DataProcessor
from data.storage import DataStorage
from api.endpoints import register_routes
from logging_setup import logger, setup_logging
from api.middleware import add_security_headers
from celery import Celery

def create_app(test_config: Optional[Dict[str, Any]] = None) -> Flask:
    """Application factory with integrated configuration"""
    setup_logging()
    
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=settings.app.secret_key,
        SWAGGER={"title": "AI Football Tracker API", "uiversion": 3},
        ENVIRONMENT=settings.app.env
    )

    # Apply configuration override for testing
    if test_config:
        app.config.update(test_config)

    # Initialize extensions
    initialize_extensions(app)
    register_routes(app)
    register_shutdown_handlers(app)
    add_security_headers(app)
    
    return app

def initialize_extensions(app: Flask) -> None:
    """Initialize Flask extensions with proper configuration"""
    SocketIO(
        app, 
        cors_allowed_origins=settings.socket.allowed_origins,
        logger=settings.socket.debug,
        engineio_logger=settings.socket.debug
    )
    
    Limiter(
        app,
        key_func=lambda: request.headers.get("X-API-Key", get_remote_address()),
        default_limits=[settings.rate_limits.default]
    )
    
    Swagger(app, template=load_swagger_spec())
    
    if settings.celery.enabled:
        initialize_celery(app)

def initialize_celery(app: Flask) -> Celery:
    """Configure and initialize Celery integration"""
    celery = Celery(
        app.name,
        broker=settings.celery.broker_url,
        backend=settings.celery.result_backend
    )
    celery.conf.update(app.config)
    return celery

def load_swagger_spec() -> Dict[str, Any]:
    """Load OpenAPI specification with current settings"""
    return {
        "info": {
            "version": settings.app.version,
            "title": "AI Football Tracker API",
            "description": settings.app.description
        },
        "host": settings.api.host,
        "basePath": settings.api.base_path
    }

def register_shutdown_handlers(app: Flask) -> None:
    """Register graceful shutdown handlers"""
    def handle_shutdown(signum, frame):
        logger.info("Received shutdown signal")
        # Add cleanup operations here
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

def initialize_services() -> None:
    """Initialize core application services"""
    try:
        services = {
            "data_collector": DataCollector(
                settings.api_keys.football_data,
                settings.api_keys.openweathermap
            ),
            "data_storage": DataStorage(
                settings.database.dsn,
                settings.aws
            ),
            "data_processor": DataProcessor(
                model_path=settings.ai.model_path,
                cache_dir=settings.ai.cache_dir
            )
        }
        logger.info("All services initialized successfully")
        return services
    except Exception as e:
        logger.critical(f"Service initialization failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    app = create_app()
    
    # Start monitoring
    start_http_server(settings.metrics.port)
    logger.info(f"Metrics server started on port {settings.metrics.port}")

    # Initialize services
    services = initialize_services()

    # Example development-mode operations
    if settings.app.env == "development":
        try:
            matches_data = services["data_collector"].fetch_football_data()
            processed_data = services["data_processor"].process_data(matches_data)
            services["data_storage"].save_to_postgres(processed_data, "matches")
        except Exception as e:
            logger.error(f"Sample data processing failed: {str(e)}")

    # Run application
    SocketIO().run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        debug=settings.server.debug,
        use_reloader=settings.server.use_reloader
    )
