from typing import Tuple, Dict, Any
from flask import request, jsonify
from flask_limiter.util import get_remote_address
from pydantic import ValidationError
from config.settings import limiter, settings
from api.schemas import PlayerData
from api.utils import logger, cache
from api.services import AnalysisService

def register_routes(app) -> None:
    """Register application routes with enhanced configuration"""
    
    @app.route('/api/v1/passing-suggestions', methods=['POST'])
    @limiter.limit(
        settings.rate_limits.passing_suggestions,  # Configured in your Settings class
        key_func=lambda: request.headers.get("X-API-Key", get_remote_address())
    )
    @cache.cached(timeout=30, query_string=True)
    def passing_suggestions() -> Tuple[Dict[str, Any], int]:
        """
        Generate AI-powered passing suggestions
        ---
        tags:
          - AI Analysis
        requestBody:
          required: true
          content:
            application/json:
              schema: PlayerData
        responses:
          200:
            description: Successfully generated suggestions
          400:
            description: Invalid input data
          429:
            description: Rate limit exceeded
          500:
            description: Internal server error
        """
        try:
            if not request.is_json:
                return jsonify({"error": "Invalid content type"}), 415
                
            payload = PlayerData.model_validate(request.get_json())
            logger.info(f"Processing request for player {payload.player_id}")
            
            suggestions = AnalysisService.generate_suggestions(payload)
            return jsonify({
                "suggestions": suggestions,
                "model_version": settings.ai_model_version
            }), 200
            
        except ValidationError as e:
            logger.warning(f"Validation error: {str(e)}")
            return jsonify({"error": "Invalid data format", "details": e.errors()}), 400
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return jsonify({"error": "Internal server error"}), 500

    @app.route('/api/v1/health')
    def health_check() -> Tuple[Dict[str, Any], int]:
        """System health check endpoint"""
        health_status = {
            "status": "healthy",
            "version": settings.app_version,
            "dependencies": {
                "database": check_db_connection(),
                "ai_model": AnalysisService.check_model_status()
            }
        }
        return jsonify(health_status), 200

def check_db_connection() -> bool:
    """Check database connectivity"""
    try:
        # Implement actual database ping
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False
