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



st.set_page_config(page_title="Stock Analysis", layout="wide")
st.title("Stock price analysis and forecasting")

stock_name = st.text_input("Enter stock ticker (e.g., AAPL):", "AAPL")

# Save charts to the project's local charts/ directory
CHART_DIRECTORY = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(CHART_DIRECTORY, exist_ok=True)

def plot_confusion_matrix(confusion_matrix, model_name):
    plt.figure(figsize=(10, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    output_path = os.path.join(CHART_DIRECTORY, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(output_path)
    plt.close()

if st.button("Fetch data and run analysis"):
    data = download_stocks(stock_name)

    if data is None or data.empty:
        st.error(f"No data found for ticker: {stock_name}")
    else:
        st.success(f"Data fetched for ticker: {stock_name}")

        st.subheader("Historical data")
        st.dataframe(data.tail())

        st.subheader("Historical price chart")
        fig, ax = plt.subplots(figsize=(6, 3))
        data['Close'].plot(ax=ax, title=f"Closing price for {stock_name}")
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

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data["Target"], mode='lines', name="Actual prices"))
        for name, preds in results.items():
            fig.add_trace(go.Scatter(x=test_data.index, y=preds, mode='lines', name=f"Forecast {name}"))

        fig.update_layout(title="Comparison of predictive models",
                          xaxis_title="Date",
                          yaxis_title="Price",
                          legend_title="Model")
        st.plotly_chart(fig)

        metrics = []
        for name, preds in results.items():
            y_true = test_data["Target"].values

            min_len = min(len(preds), len(y_true))
            mae = mean_absolute_error(y_true[:min_len], preds[:min_len])
            mse = mean_squared_error(y_true[:min_len], preds[:min_len])
            r2 = r2_score(y_true[:min_len], preds[:min_len])

            metrics.append({"Model": name, "MAE": mae, "MSE": mse, "R²": r2})

    metrics_df = pd.DataFrame(metrics)
    st.subheader("Model metrics comparison")
    st.dataframe(metrics_df)

    accuracy_scores = []
    tolerance = 10

    for name, preds in results.items():
        y_true = test_data["Target"].values

        min_len = min(len(preds), len(y_true))
        accuracy = np.mean(np.abs(y_true[:min_len] - preds[:min_len]) <= tolerance)
        accuracy_scores.append({"Model": name, "Accuracy (%)": accuracy * 100})

        accuracy_df = pd.DataFrame(accuracy_scores)
        st.subheader("Model accuracy score comparison")
        st.dataframe(accuracy_df)

        st.subheader("Model confusion matrices")
        for name, preds in results.items():
            y_true = test_data["Target"].values

            min_len = min(len(preds), len(y_true))
            rounded_preds = np.round(preds[:min_len])
            rounded_true = np.round(y_true[:min_len])

            confusion_matrix = pd.crosstab(rounded_true, rounded_preds, rownames=['Actual'], colnames=['Predicted'], margins=True)
            st.write(f"Model: {name}")
            st.dataframe(confusion_matrix)

            plot_confusion_matrix(confusion_matrix, name)


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
    st.subheader("Predicted price movement direction")
    st.dataframe(directions_df)
