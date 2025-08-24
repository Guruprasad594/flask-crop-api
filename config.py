import os
from firebase_admin import credentials, initialize_app

class Config:
    DEBUG = os.environ.get('DEBUG', 'True') == 'True'
    PROPAGATE_EXCEPTIONS = True
    FIREBASE_DB_URL = os.environ.get('FIREBASE_DB_URL', 'https://soilaid-bb736-default-rtdb.firebaseio.com/')
    FIREBASE_CRED_PATH = os.environ.get('FIREBASE_CRED_PATH', 'serviceAccountKey.json')

# Initialize Firebase with credentials from environment-configured path
def initialize_firebase_app():
    cred_path = os.environ.get('FIREBASE_CRED_PATH', 'serviceAccountKey.json')
    cred = credentials.Certificate(cred_path)
    initialize_app(cred, {
        'databaseURL': os.environ.get('FIREBASE_DB_URL')
    })

# Usage in your Flask app initialization
if __name__ == '__main__':
    initialize_firebase_app()
    from app import app  # import your Flask app instance
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=Config.DEBUG)
