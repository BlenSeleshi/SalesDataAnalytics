import torch
import logging
import numpy as np

def predict_future_sales_lstm(lstm_model, X_test_tensor, num_weeks=6):
    """
    Predict 6 weeks (42 days) into the future using the trained LSTM model.
    """
    lstm_model.eval()
    predictions = []

    for i in range(num_weeks * 7):
        # Predict the next sales value
        with torch.no_grad():
            next_pred = lstm_model(X_test_tensor.unsqueeze(0))
        
        predictions.append(next_pred.item())
        
        # Roll the test data by one time step and insert the new prediction
        X_test_tensor = torch.cat([X_test_tensor[1:], torch.tensor([[next_pred]], dtype=torch.float32)], dim=0)
        
    return np.array(predictions)
