from fastapi import FastAPI
import pandas as pd
import joblib
import os
from datetime import datetime

app = FastAPI()

# Load model & feature names
model = joblib.load("churn_model.pkl")
feature_names = joblib.load("feature_names.pkl")
csv_file = r"C:\Aparaitech\Customer_churn_system\dummy_customers.csv"
# Dummy database
csv_file = "dummy_customers.csv"

#  Monitoring & Alert Files
LOG_FILE = "prediction_logs.csv"
ALERT_FILE = "alerts.csv"
CRM_FILE = "crm_tasks.csv"


@app.get("/")
def home():
    return {"message": "Churn API Running 🚀"}


@app.get("/predict/{customer_id}")
def predict_customer(customer_id: str):
    try :
        data = pd.read_csv(csv_file)

    # Find customer
        customer = data[data['customerID'] == customer_id]

        if customer.empty:
            return {"error": "Customer not found"}
    
    # keep original for crm

        customer_original = customer.copy()
        print("step 1 ok ")
    # Drop ID column
        customer = customer.drop(columns=['customerID'])

    # ================= FEATURE ENGINEERING =================

    # avg_spend
        customer['TotalCharges'] = pd.to_numeric(customer['TotalCharges'], errors='coerce')
        customer['avg_spend'] = customer['TotalCharges'] / (customer['tenure'] + 1)

    # tenure group
        customer['tenure_group']=pd.cut(
            customer['tenure'],
            bins=[0,12,24,60,100],
            labels=['New','Mid','Loyal','Old']
        )
        print("step 2 ok")

    #  IMPORTANT: match training features
        ml_customer = pd.get_dummies(customer,drop_first=True)
    
        print("step 3 ok")
        
        ml_customer = ml_customer.reindex(columns=feature_names, fill_value=0)
    
        print("step 4 ok")

    # ==========Prediction===========
  
        prediction = model.predict(ml_customer)[0]
        probability = model.predict_proba(ml_customer)[0][1]
        print("step 5 ok")
        prediction = prediction.item() if hasattr(prediction, 'item') else prediction
        probability = probability.item() if hasattr(probability, 'item') else probability

        prediction = int(prediction)
        probability = float(probability)
        print("step 6 ok")

    # ================= MONITORING (LOGGING) =================

        log_data = pd.DataFrame({
            "customerID": [str(customer_id)],
            "prediction": [int(prediction)],
            "probability": [float(probability)],
            "timestamp": [str(datetime.now())]
        })
        if not os.path.exists(LOG_FILE):
            log_data.to_csv(LOG_FILE, index=False)
        else:
            log_data.to_csv(LOG_FILE, mode='a', header=False, index=False)

    # ================= ALERT SYSTEM =================
        if probability > 0.8:
            alert_data = pd.DataFrame({
                "customerID": [str(customer_id)],
                "risk": [float(probability)],
                "timestamp": [str(datetime.now())]
            })

            if not os.path.exists(ALERT_FILE):
                alert_data.to_csv(ALERT_FILE, index=False)
            else:
                alert_data.to_csv(ALERT_FILE, mode='a', header=False, index=False)

    # ================= CRM SIMULATION =================

        if probability > 0.7:
            action = "High Priority Call"
        else:
            action = "Email Follow-up"

        crm_data = pd.DataFrame({
            "customerID": [str(customer_id)],
            "action": [str(action)],
            "status": ["Pending"],
            "timestamp": [str(datetime.now())]
        })

        if not os.path.exists(CRM_FILE):
            crm_data.to_csv(CRM_FILE, index=False)
        else:
            crm_data.to_csv(CRM_FILE, mode='a', header=False, index=False)

        return {
            "customerID": str(customer_id),
            "prediction": int(prediction),
            "probability": float(probability),
            "avg_spend": float(customer['avg_spend'].values[0]),
            "recommended_action": action
        }
    except Exception as e:
        print("EXPECTED:", len(feature_names))
        print("GOT:", len(ml_customer.columns))

        print("🔥 ERROR:", str(e))

        return {"error": str(e)}