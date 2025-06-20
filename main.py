from data_loader import download_stocks
from visualization import plot_stock_data, add_technical_indicators
from preprocessing import preprocessing_stock_data
from model import train_model, predictions
import matplotlib.pyplot as plt
import os
from visualization import CHART_DIRECTORY
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs(CHART_DIRECTORY, exist_ok=True)

data = download_stocks("AAPL")

plot_stock_data(data, "AAPL")

add_technical_indicators(data)

data = preprocessing_stock_data(data)

train_size = int(len(data) * 0.8)

train_data = data[:train_size]

test_data = data[train_size:]



model = train_model(train_data)

predicts = predictions(model, test_data)

# Spłaszczenie predykcji

predicts = predicts.flatten()

if len(predicts) != len(test_data.index):
    raise ValueError(f"Rozmiar predictions ({len(predicts)}) nie zgadza się z rozmiarem test_data.index ({len(test_data.index)})")

# Wizualizacja

plt.figure(figsize=(10, 6))
plt.plot(test_data.index, test_data['Target'], label="Rzeczywiste ceny")
plt.plot(test_data.index, predicts, label="Prognozowane ceny")
plt.title("Predykcja cen akcji")
plt.xlabel("Data")
plt.ylabel("Cena")
plt.legend()

output_path = os.path.join(CHART_DIRECTORY, "predykcja_cen.png")

plt.savefig(output_path)

# Obliczanie metryk walidacyjnych
mae = mean_absolute_error(test_data['Target'], predicts)
mse = mean_squared_error(test_data['Target'], predicts)
r2 = r2_score(test_data['Target'], predicts)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")
