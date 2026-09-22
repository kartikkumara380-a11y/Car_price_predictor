import pandas as pd
df = pd.read_csv("data/car_prediction_data.csv")
print(df.head())
print(df.columns.tolist())
print(df.info())
print(df.head())
print(df.isnull().sum())
print(df.duplicated().sum())
df = df.drop_duplicates()
df['Car_Age'] = 2026 - df['Year']
df = df.drop('Year', axis=1)
df['Brand'] = df['Car_Name'].apply(lambda x: x.split()[0])
df = df.drop('Car_Name', axis=1)
df = pd.get_dummies(df, columns=['Brand'], drop_first=True)
df = pd.get_dummies(df, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)
print(df.head())
print(df.dtypes)
df = df.drop_duplicates()
print(df.shape)

# Remove outliers
df = df[df['Present_Price'] < 50]
df = df[df['Kms_Driven'] < 200000]
print("Shape after removing outliers:", df.shape)

# New features
df['Price_per_Age'] = df['Present_Price'] / (df['Car_Age'] + 1)
df['Price_per_Km'] = df['Present_Price'] / (df['Kms_Driven'] + 1)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X = X.astype({col: int for col in X.select_dtypes('bool').columns})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(10))

scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("CV R² scores:", scores)
print("Mean CV R²:", scores.mean())

import joblib

# Save the trained model
joblib.dump(model, 'models/car_price_model.pkl')

# Save the column names (needed later to match input format)
joblib.dump(X.columns.tolist(), 'models/model_columns.pkl')

print("Model saved successfully!")