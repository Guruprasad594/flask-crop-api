import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("crop.csv")

print("Columns detected:", df.columns.tolist())

label_encoder = LabelEncoder()
df['crop'] = label_encoder.fit_transform(df['crop'])
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph']]
y = df['crop']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\n Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
 
try:
    input_data = pd.DataFrame([[90, 40, 40, 26, 20, 6.5]],columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph'])
    prediction = model.predict(input_data)
    crop_name = label_encoder.inverse_transform(prediction)
    print("Suggested crop:", crop_name[0])
except Exception as e:
    print("Prediction error:", e)
joblib.dump(model, 'crop_prediction_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
