from dashboard import app, socketio
import pytest

@pytest.fixture
def client():
    app.config['TESTING'] = True
    return app.test_client()

def test_dashboard_loads(client):
    response = client.get('/')
    assert response.status_code == 200

def test_socket_connection():
    client = socketio.test_client(app)
    assert client.is_connected()
    received = client.get_received()
    assert len(received) > 0
