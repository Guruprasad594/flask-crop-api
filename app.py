from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and encoder
model = joblib.load('crop_prediction_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return "WELCOME TO CROP-PREDICTION MODEL"

@app.route('/predict', methods=['POST'])
def predict_crop():
    try:
        # Extract data from JSON
        data = request.get_json()
        input_data = pd.DataFrame([[
            data['N'],
            data['P'],
            data['K'],
            data['temperature'],
            data['humidity'],
            data['ph']
        ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph'])

        # Predict
        prediction = model.predict(input_data)
        crop = label_encoder.inverse_transform(prediction)

        return jsonify({
            "predicted_crop": crop[0]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
