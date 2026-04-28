import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL = pickle.load(open('Model.sav', 'rb'))

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

CATEGORY_MAPS = {
    'Dependents': {'No': 0, 'Yes': 1},
    'OnlineSecurity': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'OnlineBackup': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'DeviceProtection': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaperlessBilling': {'No': 0, 'Yes': 1},
}

FORM_OPTIONS = {
    'Dependents': list(CATEGORY_MAPS['Dependents']),
    'OnlineSecurity': list(CATEGORY_MAPS['OnlineSecurity']),
    'OnlineBackup': list(CATEGORY_MAPS['OnlineBackup']),
    'DeviceProtection': list(CATEGORY_MAPS['DeviceProtection']),
    'TechSupport': list(CATEGORY_MAPS['TechSupport']),
    'Contract': list(CATEGORY_MAPS['Contract']),
    'PaperlessBilling': list(CATEGORY_MAPS['PaperlessBilling']),
}

DEFAULT_FORM_VALUES = {
    'Dependents': 'No',
    'tenure': '12',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'MonthlyCharges': '70.35',
    'TotalCharges': '845.50',
}


def build_input_frame(form_data):
    row = {
        'Dependents': CATEGORY_MAPS['Dependents'][form_data['Dependents']],
        'tenure': float(form_data['tenure']),
        'OnlineSecurity': CATEGORY_MAPS['OnlineSecurity'][form_data['OnlineSecurity']],
        'OnlineBackup': CATEGORY_MAPS['OnlineBackup'][form_data['OnlineBackup']],
        'DeviceProtection': CATEGORY_MAPS['DeviceProtection'][form_data['DeviceProtection']],
        'TechSupport': CATEGORY_MAPS['TechSupport'][form_data['TechSupport']],
        'Contract': CATEGORY_MAPS['Contract'][form_data['Contract']],
        'PaperlessBilling': CATEGORY_MAPS['PaperlessBilling'][form_data['PaperlessBilling']],
        'MonthlyCharges': float(form_data['MonthlyCharges']),
        'TotalCharges': float(form_data['TotalCharges']),
    }
    return pd.DataFrame([row], columns=FEATURE_ORDER)


@app.route("/")
def home_page():
    return render_template(
        'home.html',
        options=FORM_OPTIONS,
        values=DEFAULT_FORM_VALUES,
    )


@app.route("/", methods=['POST'])
def predict():
    form_values = {field: request.form[field] for field in DEFAULT_FORM_VALUES}
    df = build_input_frame(form_values)
    single = MODEL.predict(df)
    probability = MODEL.predict_proba(df)[:, 1] * 100

    if single == 1:
        op1 = "This Customer is likely to be Churned!"
        op2 = f"Confidence level is {np.round(probability[0], 2)}"
    else:
        op1 = "This Customer is likely to Continue!"
        op2 = f"Confidence level is {np.round(100 - probability[0], 2)}"

    return render_template(
        "home.html",
        op1=op1,
        op2=op2,
        options=FORM_OPTIONS,
        values=form_values,
    )


if __name__ == '__main__':
    app.run(debug=False)
