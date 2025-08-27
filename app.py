from flask import Flask, jsonify, abort
import joblib
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
import datetime
import werkzeug.exceptions
from config import Config


app = Flask(__name__)
app.config.from_object(Config)


# Initialize Firebase Admin SDK
cred = credentials.Certificate(app.config['FIREBASE_CRED_PATH'])
firebase_admin.initialize_app(cred, {
    'databaseURL': app.config['FIREBASE_DB_URL']
})


# Load model and encoder
model = joblib.load('crop_prediction_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')


def fetch_latest_sensor_data():
    ref = db.reference('/sensor_data')
    data = ref.get()
    if not data:
        return None
    latest_key = sorted(data.keys())[-1]
    return data[latest_key]


# Custom error handlers for JSON responses


@app.errorhandler(400)
def handle_400_error(e):
    description = e.description if isinstance(e, werkzeug.exceptions.HTTPException) else str(e)
    return jsonify({
        "error": "Bad Request",
        "message": description,
        "status": 400
    }), 400


@app.errorhandler(404)
def handle_404_error(e):
    description = e.description if isinstance(e, werkzeug.exceptions.HTTPException) else str(e)
    return jsonify({
        "error": "Not Found",
        "message": description,
        "status": 404
    }), 404


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({
        "error": "Internal Server Error",
        "message": str(e)
    }), 500


@app.route('/')
def home():
    return "WELCOME TO CROP-PREDICTION MODEL"


@app.route('/predict', methods=['GET'])
def predict():
    sensor_data = fetch_latest_sensor_data()
    if sensor_data is None:
        abort(404, description='No sensor data found')

    input_df = pd.DataFrame([[
        sensor_data.get('N'),
        sensor_data.get('P'),
        sensor_data.get('K'),
        sensor_data.get('temperature'),
        sensor_data.get('humidity'),
        sensor_data.get('ph'),
    ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph'])

    prediction = model.predict(input_df)
    crop = label_encoder.inverse_transform(prediction)[0]

    response = {
        "predicted_crop": crop,
        "sensor_data": sensor_data
    }

    # Log prediction to database
    prediction_ref = db.reference('/predictions')
    record = {
        'timestamp': datetime.datetime.now().isoformat(),
        'sensor_data': sensor_data,
        'predicted_crop': crop,
    }
    prediction_ref.push(record)

    return jsonify(response)


@app.route('/api/soil-fertility', methods=['GET'])
def get_soil_fertility():
    sensor_data = fetch_latest_sensor_data()
    if sensor_data is None:
        abort(404, description='No sensor data found')

    response = {
        'N_level': sensor_data.get('N'),
        'P_level': sensor_data.get('P'),
        'K_level': sensor_data.get('K'),
        'pH': sensor_data.get('ph'),
        'temperature': sensor_data.get('temperature'),
        'humidity': sensor_data.get('humidity'),
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
