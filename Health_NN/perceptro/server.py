from flask import Flask, request, jsonify
import torch
import pandas as pd
import numpy as np

# Assuming the Neural.py file contains the NeuralNetwork class definition
from Neural import NeuralNetwork 

app = Flask(__name__)

# --- Load the model and setup preprocessing ---
# Make sure to adjust these paths if your file structure is different.
MODEL_PATH = 'best_model.pth'

# Define the model architecture exactly as it was when saved
input_size = 4  # Based on the columns in your smart_health_tracker_data.csv (e.g., heart_rate, stress_level, etc.)
hidden_size = 100
output_size = 1
model = NeuralNetwork(input_size, hidden_size, output_size)

try:
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: The model file '{MODEL_PATH}' was not found.")
    exit()
    
# In a real-world scenario, you would also load the scaler used for training.
# For this example, we'll assume the input data is already scaled.

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making a real-time sleep quality prediction.
    
    Expected JSON input:
    {
        "data": [heart_rate, stress_level, etc.]
    }
    """
    if request.is_json:
        try:
            data = request.get_json()
            input_features = data.get('data')

            if not isinstance(input_features, list) or len(input_features) != input_size:
                return jsonify({"error": f"Invalid input format. Expected a list of {input_size} numbers."}), 400

            # Convert to a PyTorch tensor
            input_tensor = torch.tensor([input_features], dtype=torch.float32)

            with torch.no_grad():
                # Make the prediction
                output = model(input_tensor)
                # Apply sigmoid to get a probability
                prediction = torch.sigmoid(output)
                
                # Convert the prediction to a human-readable result
                result = "Well-rested" if prediction.item() > 0.5 else "Not well-rested"

            return jsonify({
                "prediction": result,
                "probability": prediction.item()
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Request body must be JSON"}), 415

if __name__ == '__main__':
    # To run this server, use the command: python server.py
    app.run(debug=True)