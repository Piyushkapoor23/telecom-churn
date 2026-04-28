import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

CATEGORY_MAPS = {
    'Dependents': {'No': 0, 'Yes': 1},
    'OnlineSecurity': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'OnlineBackup': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'DeviceProtection': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaperlessBilling': {'No': 0, 'Yes': 1},
    'Churn': {'No': 0, 'Yes': 1},
}

FEATURE_ORDER = [
    'Dependents',
    'tenure',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'Contract',
    'PaperlessBilling',
    'MonthlyCharges',
    'TotalCharges',
]


def load_training_frame():
    df = pd.read_csv('Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    for column, mapping in CATEGORY_MAPS.items():
        df[column] = df[column].map(mapping)

    return df


def train_model():
    df = load_training_frame()
    x = df[FEATURE_ORDER]
    y = df['Churn']
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = GradientBoostingClassifier(
        criterion='squared_error',
        learning_rate=0.3,
        max_depth=19,
        max_leaf_nodes=24,
        min_samples_leaf=9,
        min_samples_split=7,
        n_estimators=150,
        random_state=42,
    )
    model.fit(x_train, y_train)

    score = accuracy_score(y_test, model.predict(x_test))
    pickle.dump(model, open('Model.sav', 'wb'))
    print(f'Saved Model.sav with test accuracy: {score:.4f}')


if __name__ == '__main__':
    train_model()
