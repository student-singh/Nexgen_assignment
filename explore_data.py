import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load your CSV
print("Loading your CSV...")
df = pd.read_csv('data/dilevery_logistics.csv')
print(f"Raw shape: {df.shape}")

# Step 2: Clean data
df['delayed'] = df['delayed'].str.lower().fillna('no')  # Standardize to lowercase
df['delivery_rating'] = pd.to_numeric(df['delivery_rating'], errors='coerce').fillna(df['delivery_rating'].median())
# Convert times: Strings like "00:00.0" to floats (0.0)
for col in ['delivery_time_hours', 'expected_time_hours']:
    df[col] = pd.to_numeric(df[col].str.replace(':', ''), errors='coerce')  # Remove :, make numeric
df = df.dropna(subset=['delivery_time_hours', 'expected_time_hours'])  # Drop bad rows
print(f"Cleaned shape: {df.shape}")

# Step 3: Derive business metrics
df['delay_hours'] = df['delivery_time_hours'] - df['expected_time_hours']
df['delay_pct'] = np.where(df['expected_time_hours'] > 0, (df['delay_hours'] / df['expected_time_hours']) * 100, 0)
# CO2: Emissions per km * distance (vehicle-specific)
def calc_co2(row):
    dist = row['distance_km']
    vtype = str(row['vehicle_type']).lower()
    if 'ev' in vtype: return 0
    elif 'bike' in vtype or 'scooter' in vtype: return 0.05 * dist
    elif 'van' in vtype: return 0.2 * dist
    elif 'truck' in vtype: return 0.5 * dist
    else: return 0.1 * dist  # Default
df['co2_kg'] = df.apply(calc_co2, axis=1)
# Opt Score: Weighted (cost 40%, delay 30%, CO2 30%) - lower is better
max_cost = df['delivery_cost'].max()
max_delay = df['delay_pct'].max() if df['delay_pct'].max() > 0 else 1
max_co2 = df['co2_kg'].max() if df['co2_kg'].max() > 0 else 1
df['opt_score'] = (
    0.4 * (df['delivery_cost'] / max_cost) +
    0.3 * (df['delay_pct'] / max_delay) +
    0.3 * (df['co2_kg'] / max_co2)
)

# Step 4: Generate insights (for problem justification)
print("\n=== KEY INSIGHTS FROM YOUR DATA ===")
print(f"Avg Delay %: {df['delay_pct'].mean():.2f}%")
print(f"Avg Cost: ${df['delivery_cost'].mean():.0f}")
print(f"Total CO2: {df['co2_kg'].sum():.0f} kg")
print("\nDelays by Weather (% Yes):")
print(df.groupby('weather_condition')['delayed'].apply(lambda x: (x == 'yes').mean() * 100).round(1))
print("\nAvg Cost by Region:")
print(df.groupby('region')['delivery_cost'].mean().round(0))
print("\nDelays by Vehicle:")
print(df[df['delayed'] == 'yes']['vehicle_type'].value_counts() if 'yes' in df['delayed'].values else "No delays yet—add some!")

# Step 5: Train ML Model (Bonus: Predict delays)
print("\n=== ML TRAINING ===")
df_ml = pd.get_dummies(df[['distance_km', 'package_weight_kg', 'weather_condition', 'region', 'vehicle_type', 'delayed']], drop_first=True)
if 'delayed_yes' in df_ml.columns:
    X = df_ml.drop('delayed_yes', axis=1)
    y = (df_ml['delayed_yes'] == 1).astype(int)
else:
    X = df_ml.drop(columns=[col for col in df_ml if 'delayed' in col.lower()], errors='ignore')
    y = (df['delayed'] == 'yes').astype(int)
if len(df) > 5:  # Min for split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Delay Prediction Accuracy: {acc:.2f}")
else:
    print("Add more rows for ML training.")

# Step 6: Save processed data
df.to_csv('data/processed_delivery_data.csv', index=False)
print(f"\nProcessed CSV saved! Rows: {len(df)}. Ready for app.")
print("\nSample processed row:")
print(df.head(1).to_string())