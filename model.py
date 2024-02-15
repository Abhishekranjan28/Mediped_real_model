import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
data = pd.read_csv("aakash data - Sheet3.csv")
print(data.columns)
# Separate features and target variable
X = data.iloc[:, 1:7]
y = data.iloc[:, -1]

# Define numeric and categorical columns
numeric_cols = ["cases", "rainfall in mm", "temperature", "avg relative humidity"]
categorical_cols = ["state", "mosquito"]

# Scale numeric columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# One-hot encode categorical columns
encoder = OneHotEncoder(sparse_output=False, drop=None)  # Change drop parameter
encoded_cols = encoder.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out())

print(X.columns,categorical_cols)

# Replace original categorical columns with encoded columns
X.drop(columns=categorical_cols, inplace=True)
X = pd.concat([X, encoded_df], axis=1)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForestClassifier
rf_classifier = RandomForestClassifier(max_depth=3, random_state=42)
rf_classifier.fit(x_train, y_train)

probability=rf_classifier.predict_proba(x_test)
print(probability)


from sklearn.metrics import accuracy_score

# Assuming 'rf' is your trained RandomForestClassifier and 'x_test' and 'y_test' are your test data
predicted_labels = rf_classifier.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predicted_labels)

print("Accuracy:", accuracy)

# Save the model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(rf_classifier, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)