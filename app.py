import streamlit as st
from data_loader import download_stocks
from preprocessing import preprocessing_stock_data
from visualization import add_technical_indicators
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import seaborn as sns
import os

# main.py wykorzystywany był do utworzenia m.in. wykresów a app.py do aplikacji webowej


st.set_page_config(page_title="Analiza akcji", layout="wide")
st.title("Analiza i prognoza cen akcji")

stock_name = st.text_input("Podaj symbol akcji (np. AAPL):", "AAPL")

CHART_DIRECTORY = "/home/adriandob2604/IOProject/charts"
os.makedirs(CHART_DIRECTORY, exist_ok=True)

# Funkcja do generowania wykresu macierzy błędów
def plot_confusion_matrix(confusion_matrix, model_name):
    plt.figure(figsize=(10, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Macierz błędów - {model_name}")
    plt.xlabel("Przewidywane")
    plt.ylabel("Rzeczywiste")
    output_path = os.path.join(CHART_DIRECTORY, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(output_path)
    plt.close()

if st.button("Pobierz dane i przeprowadź analizę"):
    data = download_stocks(stock_name)

    if data is None or data.empty:
        st.error(f"Nie znaleziono danych dla akcji: {stock_name}")
    else:
        st.success(f"Pobrano dane dla akcji: {stock_name}")

        st.subheader("Dane historyczne")
        st.dataframe(data.tail())

        st.subheader("Wykres danych historycznych")
        fig, ax = plt.subplots(figsize=(6, 3))
        data['Close'].plot(ax=ax, title=f"Cena zamknięcia dla {stock_name}")
        st.pyplot(fig, use_container_width=False)

        add_technical_indicators(data)
        data = preprocessing_stock_data(data)

        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor()
        }

        results = {}
        for name, model in models.items():
            model.fit(train_data[["Close"]], train_data["Target"])
            predictions = model.predict(test_data[["Close"]])
            results[name] = predictions

        # Wykres porównawczy
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data["Target"], mode='lines', name="Rzeczywiste ceny"))
        for name, preds in results.items():
            fig.add_trace(go.Scatter(x=test_data.index, y=preds, mode='lines', name=f"Prognoza {name}"))

        fig.update_layout(title="Porównanie modeli predykcyjnych",
                          xaxis_title="Data",
                          yaxis_title="Cena",
                          legend_title="Model")
        st.plotly_chart(fig)

        # Metryki
        metrics = []
        for name, preds in results.items():
            y_true = test_data["Target"].values

            min_len = min(len(preds), len(y_true))
            mae = mean_absolute_error(y_true[:min_len], preds[:min_len])
            mse = mean_squared_error(y_true[:min_len], preds[:min_len])
            r2 = r2_score(y_true[:min_len], preds[:min_len])

            metrics.append({"Model": name, "MAE": mae, "MSE": mse, "R²": r2})

        metrics_df = pd.DataFrame(metrics)
        st.subheader("Porównanie metryk modeli")
        st.dataframe(metrics_df)

        # Accuracy Score
        accuracy_scores = []
        tolerance = 10

        for name, preds in results.items():
            y_true = test_data["Target"].values

            min_len = min(len(preds), len(y_true))
            accuracy = np.mean(np.abs(y_true[:min_len] - preds[:min_len]) <= tolerance)
            accuracy_scores.append({"Model": name, "Accuracy (%)": accuracy * 100})

        accuracy_df = pd.DataFrame(accuracy_scores)
        st.subheader("Porównanie accuracy score modeli")
        st.dataframe(accuracy_df)

        # Macierze błędów
        st.subheader("Macierze błędów modeli")
        for name, preds in results.items():
            y_true = test_data["Target"].values

            min_len = min(len(preds), len(y_true))
            rounded_preds = np.round(preds[:min_len])
            rounded_true = np.round(y_true[:min_len])

            confusion_matrix = pd.crosstab(rounded_true, rounded_preds, rownames=['Rzeczywiste'], colnames=['Przewidywane'], margins=True)
            st.write(f"Model: {name}")
            st.dataframe(confusion_matrix)

            plot_confusion_matrix(confusion_matrix, name)

        # Funkcja do określenia kierunku zmiany ceny

        def predict_direction(current_price, predicted_price):
            if predicted_price > current_price:
                return "buy"
            elif predicted_price < current_price:
                return "sell"
            else:
                return "hold"

        directions = []
        for name, preds in results.items():
            y_true = test_data["Target"].values

            min_len = min(len(preds), len(y_true))
            for i in range(min_len):
                direction = predict_direction(y_true[i], preds[i])
                directions.append({"Model": name, "Date": test_data.index[i], "Direction": direction})

        directions_df = pd.DataFrame(directions)
        st.subheader("Predykcja kierunku zmiany ceny")
        st.dataframe(directions_df)
