def preprocessing_stock_data(data):
    data["Target"] = data["Close"].shift(-1)
    data = data.dropna()
    return data
