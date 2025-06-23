import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from same3 import LSTMModel
from same3 import LSTMModel
import torch.nn.functional as F
import numpy as np
# Model parameters (update these with your model specifics)
input_size = 9   # The correct input size used during training
hidden_size = 64  # The correct hidden size used during training
num_classes = 15  # The correct number of classes for the output layer

# Create an instance of the model
model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

class_labels = ['BENIGN', 'DoS Hulk', 'FTP-Patator', 'PortScan', 'DDoS',
                'DoS Slowhttptest', 'DoS slowloris', 'Web Attack – XSS', 'Bot',
                'DoS GoldenEye', 'SSH-Patator', 'Web Attack – Brute Force',
                'Infiltration', 'Web Attack – SQL Injection', 'Heartbleed']

def predict_from_netcsv(csv_file, model=model, class_labels=class_labels):
    """
    Load a CSV file, preprocess the data, and make predictions using the provided model.

    Args:
        csv_file (str): Path to the CSV file with input features.
        model (torch.nn.Module): Trained LSTM model for predictions.
        class_labels (list): List of class labels (for decoding the model output).

    Returns:
        str: Predicted class label.
    """
    # Step 1: Load the CSV file
    print(f"Loading data from {csv_file}...")
    data = pd.read_csv(csv_file)

    # Step 2: Preprocess the input data
    data = data.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric data to NaN
    input_data = torch.tensor(data.values, dtype=torch.float32)

    # Reshape the input for LSTM: (batch_size, sequence_length, input_size)
    input_data = input_data.unsqueeze(1)  # Add sequence dimension (samples x 1 x features)

    # Ensure the input has 3 dimensions (batch_size, sequence_length, input_size)
    if input_data.dim() != 3:
        raise ValueError(f"Expected input to have 3 dimensions, got {input_data.dim()}D instead")

    # Step 3: Make predictions using the model
    with torch.no_grad():  # Disable gradient computation for inference
        predictions = model(input_data)

    # Step 4: Apply softmax to get probabilities for each class
    probabilities = F.softmax(predictions, dim=1)

    # Get the index of the predicted class
    predicted_class_index = torch.argmax(probabilities, dim=1).item()

    # Step 5: Decode the class index to get the class label
    label_encoder = LabelEncoder()
    label_encoder.fit(class_labels)
    model.load_state_dict(torch.load('net_model.pth'))
    model.eval()
    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]

    # Return the predicted class label
    return predicted_class

# Example usage:

class_labels = ['BENIGN', 'DoS Hulk', 'FTP-Patator', 'PortScan', 'DDoS',
                'DoS Slowhttptest', 'DoS slowloris', 'Web Attack – XSS', 'Bot',
                'DoS GoldenEye', 'SSH-Patator', 'Web Attack – Brute Force',
                'Infiltration', 'Web Attack – SQL Injection', 'Heartbleed']

# Model parameters (update these with your model specifics)

# Predict from CSV
csv_file = "network_test.csv"  # Replace with your actual CSV file path
predicted_class = predict_from_netcsv(csv_file)

# Print the result
print(f"Predicted class: {predicted_class}")