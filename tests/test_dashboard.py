from dashboard import app, socketio
import pytest

@pytest.fixture(scope='module')
def test_client():
    """Create a test client for the Flask application."""
    # Configure test environment
    app.config.update({
        'TESTING': True,
        'WTF_CSRF_ENABLED': False,  # Disable CSRF protection for testing
        'SERVER_NAME': 'localhost.localdomain'  # Required for socket.IO testing
    })
    
    # Establish application context
    with app.app_context():
        yield app.test_client()

@pytest.fixture
def socket_client():
    """Create a Socket.IO test client with automatic cleanup."""
    with socketio.test_client(app, namespace='/dashboard') as client:
        yield client
        client.disconnect()

def test_dashboard_loads(test_client):
    """Test that the dashboard root route returns expected content."""
    response = test_client.get('/')
    
    # Basic response validation
    assert response.status_code == 200
    assert response.content_type == 'text/html; charset=utf-8'
    
    # Content validation
    assert b'<title' in response.data
    assert b'<div id="root">' in response.data  # Verify main content container
    assert b'<script src="/static/' in response.data  # Check for frontend assets

def test_socket_connection(socket_client):
    """Test WebSocket connection establishment and initial handshake."""
    # Connection status check
    assert socket_client.is_connected()
    
    # Verify namespace connection
    assert '/dashboard' in socket_client.namespaces
    
    # Check initial message payload
    received = socket_client.get_received('/dashboard')
    assert len(received) == 1
    
    # Validate initial connection message format
    connection_message = received[0]
    assert connection_message['name'] == 'connection_ack'
    assert 'status' in connection_message['args'][0]
    assert 'message' in connection_message['args'][0]
    assert connection_message['args'][0]['status'] == 'success'
