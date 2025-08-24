from flask import Flask, request, jsonify
import joblib
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
import datetime
import os

from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Initialize Firebase Admin SDK using config values
cred = credentials.Certificate(app.config['FIREBASE_CRED_PATH'])
firebase_admin.initialize_app(cred, {
    'databaseURL': app.config['FIREBASE_DB_URL']
})

# Load model and encoder
model = joblib.load('crop_prediction_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load crop dataset and calculate avg nutrients
df = pd.read_csv('crop.csv')
avg_nutrients = df.groupby('crop')[['N', 'P', 'K']].mean().reset_index()
avg_nutrients.rename(columns={'N': 'avg_N', 'P': 'avg_P', 'K': 'avg_K'}, inplace=True)


def suggest_fertilizer_for_crop(chosen_crop, soil_N, soil_P, soil_K, avg_nutrients_df, threshold=10):
    crop_data = avg_nutrients_df[avg_nutrients_df['crop'] == chosen_crop]
    if crop_data.empty:
        return [f"Crop '{chosen_crop}' not found in dataset."]

    required_N = float(crop_data['avg_N'].iloc[0])
    required_P = float(crop_data['avg_P'].iloc[0])
    required_K = float(crop_data['avg_K'].iloc[0])

    add_N = required_N - soil_N
    add_P = required_P - soil_P
    add_K = required_K - soil_K

    suggestions = []
    need_fertilizer = False

    if add_N > threshold:
        suggestions.append(f"Add {add_N:.2f} units of Nitrogen (N)")
        need_fertilizer = True
    if add_P > threshold:
        suggestions.append(f"Add {add_P:.2f} units of Phosphorus (P)")
        need_fertilizer = True
    if add_K > threshold:
        suggestions.append(f"Add {add_K:.2f} units of Potassium (K)")
        need_fertilizer = True

    if not need_fertilizer:
        suggestions.append("All nutrient levels are adequate")

    return suggestions


def fetch_latest_sensor_data():
    ref = db.reference('/sensor_data')
    data = ref.get()
    if not data:
        return None
    latest_key = sorted(data.keys())[-1]
    return data[latest_key]


@app.route('/')
def home():
    return "WELCOME TO CROP-PREDICTION MODEL"


@app.route('/predict', methods=['POST'])
def predict_crop():
    try:
        data = request.get_json()
        N = data.get('N')
        P = data.get('P')
        K = data.get('K')
        temperature = data.get('temperature')
        humidity = data.get('humidity')
        ph = data.get('ph')
        farmer_crop = data.get('farmer_crop')

        # Basic validation
        if None in [N, P, K, temperature, humidity, ph]:
            return jsonify({"error": "Missing one of the required fields"}), 400

        input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph]],
                                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph'])
        prediction = model.predict(input_df)
        crop = label_encoder.inverse_transform(prediction)[0]

        chosen_crop = farmer_crop if farmer_crop else crop

        fertilizer_suggestions = suggest_fertilizer_for_crop(
            chosen_crop, N, P, K, avg_nutrients, threshold=10
        )

        return jsonify({
            "predicted_crop": crop,
            "farmer_chosen_crop": chosen_crop,
            "fertilizer_suggestions": fertilizer_suggestions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/auto_predict', methods=['GET'])
def auto_predict():
    sensor_data = fetch_latest_sensor_data()
    if sensor_data is None:
        return jsonify({'error': 'No sensor data found'}), 404

    input_data = {
        'N': sensor_data.get('N'),
        'P': sensor_data.get('P'),
        'K': sensor_data.get('K'),
        'temperature': sensor_data.get('temperature'),
        'humidity': sensor_data.get('humidity'),
        'ph': sensor_data.get('ph'),  # Ensure key names match sensor data
        'farmer_crop': None
    }

    with app.test_request_context('/predict', method='POST', json=input_data):
        response = predict_crop()
        if isinstance(response, tuple):
            resp_obj = response[0]
        else:
            resp_obj = response
        prediction_response = resp_obj.get_json()

    prediction_ref = db.reference('/predictions')
    record = {
        'timestamp': datetime.datetime.now().isoformat(),
        'sensor_data': sensor_data,
        'predicted_crop': prediction_response.get('predicted_crop'),
        'fertilizer_suggestions': prediction_response.get('fertilizer_suggestions')
    }
    prediction_ref.push(record)

    return jsonify(prediction_response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
