import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt


# RandomForestRegressor is a powerful non-linear regression model, real world, noisy datasets
# This code builds a machine learning model to predict the posted waiting time (SPOSTMIN) for Disney attractions
# using time and metadata features (like hour, minute, weekday, holidays, and attraction name)

# Load data
df = pd.read_csv("../disney-world/data/all_waiting_times.csv")
metadata = pd.read_csv("../disney-world/data/overview data/metadata.csv")

# Preprocess
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['date'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute
df['weekday'] = df['datetime'].dt.day_name()

# Merge HOLIDAYM
metadata['DATE'] = pd.to_datetime(metadata['DATE']).dt.date
df = df.merge(metadata[['DATE', 'HOLIDAYM']], left_on='date', right_on='DATE', how='left')
df['HOLIDAYM'] = df['HOLIDAYM'].fillna(0)

# Filter invalid
df = df[(df['SPOSTMIN'].notna()) & (df['SPOSTMIN'] >= 0) & (df['SPOSTMIN'] < 300)]

# Train model
X = df[['hour', 'minute', 'weekday', 'HOLIDAYM', 'attraction']]
y = df['SPOSTMIN']
X_encoded = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=10, random_state=5)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual SPOSTMIN")
plt.ylabel("Predicted SPOSTMIN")
plt.title("Predicted vs Actual Posted Wait Times")
plt.plot([0, 300], [0, 300], color='red', linestyle='--')
plt.show()

# Python-specific file format used to save and load objects
# Save
joblib.dump(model, "wait_time_model.pkl")
joblib.dump(X_encoded.columns.tolist(), "model_features.pkl")