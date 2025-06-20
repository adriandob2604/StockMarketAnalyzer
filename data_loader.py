import yfinance as yf
def download_stocks(stockname):
    stock = yf.Ticker(stockname)
    data = stock.history(period="6mo") 
    if data is None or data.empty:
        print(f"Nie znaleziono danych dla akcji: ${stockname}")
    return data
