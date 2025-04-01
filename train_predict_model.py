import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# RandomForestRegressor is a powerful non-linear regression model, real world,
# noisy datasets (park data is messy and unpredictable)
# we can handel many features
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

# Data Is Non-Linear Before Training
# As the hour increases, the wait time increases or decreases in a straight, predictable way (like a line).
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='hour', y='SPOSTMIN', alpha=0.1)
plt.title("Wait Time vs Hour of Day")
plt.xlabel("Hour")
plt.ylabel("SPOSTMIN (Posted Wait Time)")
plt.grid(True)
plt.show()

# Wait Time by Weekday (Boxplot) noisy datasets
# This boxplot clearly shows that our dataset is noisy. Every weekday has a wide range of wait times, with many extreme outliers, and no consistent pattern.
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='weekday', y='SPOSTMIN', order=[
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])
plt.title("Distribution of Wait Times per Weekday")
plt.xlabel("Weekday")
plt.ylabel("SPOSTMIN")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Attractions: huge variation ride to ride
# Ride wait times are not equal – some are long and variable, others short and predictable
# There's no one-size-fits-all pattern, which adds complexity
# Our model must learn individual ride behavior
top_attractions = df['attraction'].value_counts().head(10).index.tolist()
df_top = df[df['attraction'].isin(top_attractions)]

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_top, x='attraction', y='SPOSTMIN')
plt.title("Wait Time Distribution for Top Attractions")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Train model
X = df[['hour', 'minute', 'weekday', 'HOLIDAYM', 'attraction']]
y = df['SPOSTMIN']
X_encoded = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=50, random_state=5)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred)) # 12.43 minutes
print("R2 Score:", r2_score(y_test, y_pred)) # 0.65
# That means our model predicts with an average error of ~12 minutes, and explains 65% of the variance.

#Visualize Model Performance
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