import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Load the dataset
data = pd.read_csv("vehicle_sensor_data.csv")

# 2. Encode the target labels
le = LabelEncoder()
data['fault_type_encoded'] = le.fit_transform(data['fault_type'])

# 3. Prepare features & target
X = data[['speed', 'engine_temp', 'oil_pressure', 'vibration', 'brake_pressure', 'gear_position']]
y = data['fault_type_encoded']

# 4. Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. Save the trained model and label encoder
with open("rf_model.pkl", "wb") as file:
    pickle.dump({'model': model, 'label_encoder': le}, file)

print("Model and label encoder saved to rf_model.pkl")
