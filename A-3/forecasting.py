import numpy as np

def ARIMA_Forecast(input_series: list, P: int, D: int, Q: int, prediction_count: int) -> list:
    # Convert the input series to a numpy array
    input_series = np.array(input_series)
    
    # Initialize the predicted series with the same values as the input series
    predicted_series = input_series.copy()
    
    # Initialize the coefficients
    phi = np.zeros(P)
    theta = np.zeros(Q)
    
    # Iterate over the number of predictions
    for i in range(prediction_count):
        # Calculate the difference series
        if len(predicted_series) >= D:
            diff_series = np.diff(predicted_series, n=D)
        else:
            diff_series = np.zeros(1)
        
        # Calculate the autoregressive terms
        for j in range(P):
            if len(predicted_series) >= P:
                predicted_series = predicted_series.reshape(-1, 1)
                phi[j] = np.corrcoef(predicted_series[D-j:-i-D, :16], predicted_series[D-i-1:-1-i, :16])[0, 1]

                #phi[j] = np.corrcoef(predicted_series[D-j:-i-D], predicted_series[D-i-1:-1-i])[0, 1]
            
        # Calculate the moving average terms
        for k in range(Q):
            if len(diff_series) >= Q:
                theta[k] = np.corrcoef(diff_series[-k-1:-1], predicted_series[D-k-1-i:-1-i])[0, 1]
                
        # Calculate the next term in the series
        next_term = 0
        for j in range(P):
            if len(predicted_series) >= P:
                next_term += phi[j] * predicted_series[-j-1-D-i]
                
        for k in range(Q):
            if len(diff_series) >= Q:
                next_term += theta[k] * diff_series[-k-1]
                
        # Add the next term to the predicted series
        predicted_series = np.append(predicted_series, next_term)
        
    # Return the new terms in the predicted sequence
    return predicted_series[-prediction_count:]
    

def HoltWinter_Forecast(input: list, alpha: float, beta: float, gamma: float, seasonality: int, prediction_count: int) -> list:
    # Initialize the level, trend, and seasonal components of the model
    level = sum(input[:seasonality]) / seasonality
    trend = (input[seasonality] - input[0]) / seasonality
    seasonal_components = [input[i] - level for i in range(seasonality)]
    
    # Iterate over the input series to generate predictions
    predictions = []
    for i in range(len(input) + prediction_count):
        # Generate a new forecast for the current time step
        forecast = level + trend + seasonal_components[i % seasonality]
        predictions.append(forecast)
        
        # Update the level, trend, and seasonal components of the model
        if i < len(input):
            obs = input[i]
            level = alpha * (obs - seasonal_components[i % seasonality]) + (1 - alpha) * level
            trend = beta * (level - sum(input[i:i+seasonality])/seasonality) + (1 - beta) * trend
            seasonal_components[i % seasonality] = gamma * (obs - level) + (1 - gamma) * seasonal_components[i % seasonality]
    
    # Return the new terms in the predicted sequence
    return predictions[-prediction_count:]



def ARIMA_Parameters(input_series:list)->tuple: # (P, D, Q)
    # Convert the input series to a numpy array
    input_series = np.array(input_series)
    
    # Initialize the best parameters as None and the best AIC score as infinity
    best_params = None
    best_aic = np.inf
    
    # Iterate over all possible combinations of (P, D, Q)
    for p in range(3):
        for d in range(3):
            for q in range(3):
                try:
                    # Fit the ARIMA model with the current combination of (P, D, Q)
                    model = ARIMA(input_series, order=(p, d, q))
                    model_fit = model.fit()
                    
                    # Calculate the AIC score of the model
                    aic = model_fit.aic
                    
                    # Update the best parameters and best AIC score if the current model has a lower AIC score
                    if aic < best_aic:
                        best_params = (p, d, q)
                        best_aic = aic
                except:
                    continue
    
    # Return the best parameters as a tuple
    return best_params


def HoltWinter_Parameters(input_series:list)->tuple: # (Alpha, Beta, Gamma, Seasonality)
    # Convert the input series to a numpy array
    input_series = np.array(input_series)
    
    # Initialize the best parameters as None and the best RMSE score as infinity
    best_params = None
    best_rmse = np.inf
    
    # Iterate over all possible combinations of alpha, beta, and gamma
    for alpha in np.arange(0, 1, 0.1):
        for beta in np.arange(0, 1, 0.1):
            for gamma in np.arange(0, 1, 0.1):
                try:
                    # Fit the Holt-Winter's model with the current combination of alpha, beta, and gamma
                    model = ExponentialSmoothing(input_series, seasonal_periods=seasonality, 
                                                           trend='add', seasonal='add').fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)

                    # Generate predictions using the model
                    predictions = model.predict(start=len(input_series), end=len(input_series) + 19)

                    # Compute the root mean squared error (RMSE) between the predicted values and the true values
                    rmse = np.sqrt(np.mean((predictions - input_series[-20:]) ** 2))

                    # If the current RMSE is better than the previous best RMSE, update the best parameters
                    if rmse < best_rmse:
                        best_params = (alpha, beta, gamma, seasonality)
                        best_rmse = rmse
                except:
                    continue
    # Return the best parameters
    return best_params





