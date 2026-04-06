from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

data = joblib.load(os.path.join(BASE_DIR, 'model', 'kmeans_model.pkl'))
model = data["model"]
scaler = data["scaler"]
centers = data["centers"]

df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'Customer_Data.csv'))

def get_label(cluster_id):
    center = centers[cluster_id]

    balance_c = center[0]
    purchases_c = center[1]
    
    if balance_c < -0.5 and purchases_c < -0.5:
        return "😴 Пассивный клиент"
    
    if balance_c > 1 and purchases_c > 1:
        return "👑 VIP клиент"
    
    if purchases_c > 0.5:
        return "💳 Активный клиент"
    
    if balance_c > 0.5:
        return "💼 Перспективный клиент"
    
    return "🌿 Стабильный клиент"


@app.route('/', methods=['GET', 'POST'])
def home():
    result = None

    if request.method == 'POST':
        try:
            balance = float(request.form['balance'])
            purchases = float(request.form['purchases'])

            data_input = np.array([[balance, purchases]])
            data_scaled = scaler.transform(data_input)

            cluster = model.predict(data_scaled)[0]

            result = get_label(cluster)

        except:
            result = "Введите только числа!"

    return render_template(
        'index.html',
        result=result,
        tables=df.head(10).to_html(),
        df=df
    )

if __name__ == '__main__':
    app.run(debug=True)