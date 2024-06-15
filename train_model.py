import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def preprocess_data():
    df = pd.read_csv('Telco-Customer-Churn.csv')

    # Drop customerID column
    df = df.drop('customerID', axis=1)

    # Convert categorical columns to dummy variables
    df = pd.get_dummies(df, drop_first=True)

    # Features and target variable
    X = df.drop('Churn_Yes', axis=1)
    y = df['Churn_Yes']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model():
    X_train, _, y_train, _ = preprocess_data()  # Fix here to include y_train

    # Train a RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'churn_model.pkl')

    # Save the model columns
    joblib.dump(X_train.columns.tolist(), 'model_columns.pkl')

    print("Model trained and saved successfully.")

if __name__ == '__main__':
    train_model()
