import joblib
import numpy as np

data = joblib.load("model/kmeans_model.pkl")

model = data["model"]   
centers = data["centers"]

balance = float(input("Balance: "))
purchases = float(input("Purchases: "))

scaler = data["scaler"]

input_scaled = scaler.transform([[balance, purchases]])
cluster = model.predict(input_scaled)[0]

balance_c, purchases_c = centers[cluster]

if balance_c < -0.5 and purchases_c < -0.5:
    label = "😴 Пассивный клиент"
elif balance_c > 1 and purchases_c > 1:
    label = "👑 VIP клиент"
elif purchases_c > 0.5:
    label = "💳 Активный клиент"
elif balance_c > 0.5:
    label = "💼 Перспективный клиент"
else:
    label = "🌿 Стабильный клиент"

print("Cluster:", cluster)
print("Result:", label)