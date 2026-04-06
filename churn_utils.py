import pandas as pd

def predict_churn(new_data, model, feature_names):
    """
    Predict churn for new data.

    Parameters:
    - new_data: DataFrame with user input features
    - model: trained ML model
    - feature_names: list of features used during model training

    Returns:
    - prediction: 0 or 1
    - probability: probability of churn
    """
    # Ensure all columns exist
    df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    for col in new_data.columns:
        if col in df.columns:
            df[col] = new_data[col]
    
    prediction = model.predict(df)
    probability = model.predict_proba(df)[:,1]
    
    return prediction, probability