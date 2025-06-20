import matplotlib.pyplot as plt
import os

CHART_DIRECTORY = "/home/adriandob2604/IOProject/charts"


# Wizualizacja kursów zamknięcia cen akcji


def plot_stock_data(data, stockname):
    
    
    data['Close'].plot(figsize=(10, 6), title=f"Cena zamknięcia dla {stockname}")
    plt.xlabel("Data")
    
    plt.ylabel("Cena zamknięcia")
    
    output_path = os.path.join(CHART_DIRECTORY, "stockdata.png")

    plt.savefig(output_path)

# Wskaźniki techniczne


def add_technical_indicators(data):

    # Średnie kroczące

    data["SMA_20"] = data["Close"].rolling(window=20).mean()

    data["SMA_50"] = data["Close"].rolling(window=50).mean()

    plt.figure(figsize=(10,6))
    plt.plot(data.index, data["Close"], label="Cena zamknięcia")
    plt.plot(data.index, data["SMA_20"], label="SMA 20-dniowa")
    plt.plot(data.index, data["SMA_50"], label="SMA 50-dniowa")

    plt.title("Średnia krocząca")
    plt.xlabel("Data")
    plt.ylabel("Cena")
    plt.legend()

    output_path = os.path.join(CHART_DIRECTORY, "srednie_kroczace.png")
    plt.savefig(output_path)


def add_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data