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

def load_model(model_name):
    """
    Loads the serialized model from disk.
    """
    logging.info(f"Loading {model_name}.")
    model = joblib.load(f"{model_name}.pkl")
    logging.info(f"{model_name} loaded successfully.")
    return model