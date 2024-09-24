import logging
import joblib
from datetime import datetime

def serialize_model(model, model_name="model"):
    """
    Serialize the trained model and save it with a timestamp.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_filename = f"{model_name}-{timestamp}.pkl"
    
    logging.info(f"Serializing {model_name} and saving as {model_filename}.")
    
    joblib.dump(model, model_filename)
    
    logging.info(f"Model {model_name} saved successfully.")
    return model_filename
