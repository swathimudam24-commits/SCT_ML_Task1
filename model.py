# ============================================================
#   TASK 01 - House Price Prediction using Linear Regression
# ============================================================

# STEP 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# STEP 2: Load Dataset
# ============================================================
df = pd.read_csv('house_price_dataset.csv')

print("=== Dataset Info ===")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nBasic Stats:\n{df.describe()}")

# ============================================================
# STEP 3: Define Features and Target
# ============================================================
X = df[['square_footage', 'bedrooms', 'bathrooms']]
y = df['price']  # in Rupees

# ============================================================
# STEP 4: Split Data - Train (80%) and Test (20%)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ============================================================
# STEP 5: Feature Scaling
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ============================================================
# STEP 6: Train the Linear Regression Model
# ============================================================
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\n=== Model Coefficients ===")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: Rs.{coef:,.2f}")
print(f"  Intercept: Rs.{model.intercept_:,.2f}")

# ============================================================
# STEP 7: Make Predictions
# ============================================================
y_pred = model.predict(X_test_scaled)

# ============================================================
# STEP 8: Evaluate the Model
# ============================================================
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n=== Model Performance ===")
print(f"  MAE  (Mean Absolute Error):       Rs.{mae:,.2f}")
print(f"  MSE  (Mean Squared Error):        Rs.{mse:,.2f}")
print(f"  RMSE (Root Mean Squared Error):   Rs.{rmse:,.2f}")
print(f"  R2   (R-Squared Score):           {r2:.4f}")

# ============================================================
# STEP 9: Plot Actual vs Predicted (in Lakhs for clean display)
# ============================================================
y_test_L = y_test / 100000
y_pred_L = y_pred / 100000

plt.figure(figsize=(8, 6))
plt.scatter(y_test_L, y_pred_L, color='steelblue', alpha=0.6, edgecolors='white', linewidth=0.3)

min_val = min(y_test_L.min(), y_pred_L.min())
max_val = max(y_test_L.max(), y_pred_L.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

plt.xlabel('Actual Prices (Rs. in Lakhs)', fontsize=13)
plt.ylabel('Predicted Prices (Rs. in Lakhs)', fontsize=13)
plt.title('Actual vs Predicted House Prices', fontsize=15)
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150)
plt.show()
print("\nGraph saved as 'actual_vs_predicted.png'")

# ============================================================
# STEP 10: Sample Predictions
# ============================================================
print("\n=== Sample Predictions ===")
sample = pd.DataFrame({
    'square_footage': [1500, 2500, 3500],
    'bedrooms':       [2,    3,    4],
    'bathrooms':      [1,    2,    3]
})
sample_scaled = scaler.transform(sample)
sample_pred   = model.predict(sample_scaled)

for i, row in sample.iterrows():
    print(f"  {row['square_footage']} sqft, {int(row['bedrooms'])} bed, "
          f"{int(row['bathrooms'])} bath => Rs.{sample_pred[i]:,.0f} "
          f"({sample_pred[i]/100000:.2f} Lakhs)")
