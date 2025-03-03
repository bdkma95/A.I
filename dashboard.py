from flask import Flask, render_template
from flask_socketio import SocketIO
import logging

logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/")
def dashboard():
    return render_template("dashboard.html")  # Create a corresponding dashboard.html

@socketio.on("connect")
def handle_connect():
    socketio.emit("update_metrics", {"tweets_engaged": 100, "airdrops_sent": 10})

if __name__ == "__main__":
    socketio.run(app, port=5000)
