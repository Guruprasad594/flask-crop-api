from flask import Flask, jsonify, abort, request
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

# Load model, encoder, and nutrient reference
model = joblib.load('crop_prediction_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
reference_df = pd.read_csv('crop.csv')

# Fertilizer nutrient contents (%)
fertilizer_contents = {
    'urea': {'N': 46},
    'ssp': {'P': 16},
    'mop': {'K': 60}
}

# Conversion factor hectare to acre
HA_TO_ACRE = 0.4047


def fetch_latest_sensor_data():
    ref = db.reference('/sensor_data')
    data = ref.get()
    if not data:
        return None
    latest_key = sorted(data.keys())[-1]
    return data[latest_key]


def get_ref_nutrients(crop):
    row = reference_df[reference_df['crop'] == crop]
    if row.empty:
        return None
    return {'N': row['N'].values[0], 'P': row['P'].values[0], 'K': row['K'].values[0]}


def suggest_fertilizer(crop, soil_nutrients):
    ref_nutrients = get_ref_nutrients(crop)
    if ref_nutrients is None:
        return f"No nutrient reference found for crop: {crop}"
    deficiency = {n: max(ref_nutrients[n] - soil_nutrients.get(n, 0), 0) for n in ['N', 'P', 'K']}
    fertilizer_amounts_ha = {
        'urea': deficiency['N'] / fertilizer_contents['urea']['N'] * 100,
        'Super Phosphate': deficiency['P'] / fertilizer_contents['ssp']['P'] * 100,
        'Potash': deficiency['K'] / fertilizer_contents['mop']['K'] * 100,
    }
    fertilizer_amounts_acre = {k: round(v * HA_TO_ACRE, 2) for k, v in fertilizer_amounts_ha.items()}
    return fertilizer_amounts_acre


# Error Handlers

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


# Routes

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


@app.route('/fertilizer-suggestion', methods=['POST'])
def fertilizer_suggestion():
    data = request.json
    crop = data.get('crop')
    if not crop:
        abort(400, description="Crop name is required")

    sensor_data = fetch_latest_sensor_data()
    if sensor_data is None:
        abort(404, description="No sensor data found")

    soil_nutrients = {
        'N': sensor_data.get('N', 0),
        'P': sensor_data.get('P', 0),
        'K': sensor_data.get('K', 0),
    }

    suggestions = suggest_fertilizer(crop, soil_nutrients)
    if isinstance(suggestions, str):
        return jsonify({"error": suggestions}), 404

    formatted_suggestions = []
    for fert, amount in suggestions.items():
        formatted_suggestions.append(f"{fert.upper()}: {amount} kg/acre")

    return jsonify({
        "crop": crop,
        "fertilizer_suggestions": formatted_suggestions
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
