from flask import Flask, request, jsonify, abort
import joblib
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
import datetime
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Initialize Firebase Admin SDK
cred = credentials.Certificate(app.config['FIREBASE_CRED_PATH'])
firebase_admin.initialize_app(cred, {
    'databaseURL': app.config['FIREBASE_DB_URL']
})

# Load crop nutrient averages
df = pd.read_csv('crop.csv')
avg_nutrients = df.groupby('crop')[['N', 'P', 'K']].mean().reset_index()
avg_nutrients.rename(columns={'N': 'avg_N', 'P': 'avg_P', 'K': 'avg_K'}, inplace=True)

def suggest_fertilizer_for_crop(chosen_crop, soil_N, soil_P, soil_K, avg_nutrients_df, threshold=10):
    crop_data = avg_nutrients_df[avg_nutrients_df['crop'] == chosen_crop]
    if crop_data.empty:
        abort(400, description=f"Crop '{chosen_crop}' not found in dataset.")

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

@app.route('/fertilizer_suggestion', methods=['POST'])
def fertilizer_suggestion():
    data = request.get_json()
    if not data:
        abort(400, description="Request JSON body required.")

    chosen_crop = data.get('crop')
    soil_N = data.get('N')
    soil_P = data.get('P')
    soil_K = data.get('K')

    if not chosen_crop:
        abort(400, description="Crop name is required in request.")
    if soil_N is None or soil_P is None or soil_K is None:
        abort(400, description="Soil nutrient data (N,P,K) are required.")

    suggestions = suggest_fertilizer_for_crop(chosen_crop.strip().lower(), soil_N, soil_P, soil_K, avg_nutrients)

    response = {
        "crop": chosen_crop,
        "fertilizer_suggestions": suggestions,
        "timestamp": datetime.datetime.now().isoformat()
    }

    # Optional: log prediction data to Firebase
    prediction_ref = db.reference('/predictions')
    prediction_ref.push({
        'timestamp': response['timestamp'],
        'crop': chosen_crop,
        'soil_nutrients': {'N': soil_N, 'P': soil_P, 'K': soil_K},
        'fertilizer_suggestions': suggestions
    })

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
