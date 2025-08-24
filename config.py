import os

class Config:
    DEBUG = os.environ.get('DEBUG', 'True') == 'True'
    PROPAGATE_EXCEPTIONS = True
    FIREBASE_DB_URL = os.environ.get('FIREBASE_DB_URL', 'https://soilaid-bb736-default-rtdb.firebaseio.com/')
    FIREBASE_CRED_PATH = os.environ.get('FIREBASE_CRED_PATH', 'serviceAccountKey.json')
