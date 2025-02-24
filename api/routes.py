from flask import jsonify
from flask_limiter.util import get_remote_address
from config.settings import limiter
from api.schemas import PlayerData

def register_routes(app):
    @app.route('/passing-suggestions', methods=['POST'])
    @limiter.limit("10 per minute")
    def passing_suggestions():
        try:
            data = PlayerData(**request.json)
            return jsonify({"message": "Improve passing accuracy under pressure."})
        except ValidationError as e:
            return jsonify({"error": str(e)}), 400

    @app.route('/health')
    def health_check():
        return jsonify({"status": "healthy"}), 200
