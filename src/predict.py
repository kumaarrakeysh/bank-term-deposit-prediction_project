import joblib
import pandas as pd

model = joblib.load('models/best_model.pkl')

def predict(data_dict):
    df = pd.DataFrame([data_dict])
    prediction = model.predict(df)
    return prediction[0]