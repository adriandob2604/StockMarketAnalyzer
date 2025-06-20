from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np

def train_model(train_data):
    
    # Dane wejściowe

    X_train = np.array(train_data["Close"]).reshape(-1, 1)
    y_train = np.array(train_data["Target"])

    # model = LinearRegression()

    # model = RandomForestRegressor()
    
    model = GradientBoostingRegressor()

    model.fit(X_train, y_train)

    return model

def predictions(model, test_data):
    X_test = np.array(test_data["Close"]).reshape(-1,1)
    predictions = model.predict(X_test)

    return predictions
