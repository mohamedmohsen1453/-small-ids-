import torch
import pandas as pd
from same import AdvancedNNModel  # Import the model class

def predict_from_csv(csv_file, pth_file):
    """
    Load a CSV file and a saved PTH model file, then predict the outputs.

    Args:
        csv_file (str): Path to the CSV file with input features.
        pth_file (str): Path to the saved PTH file (model weights).

    Returns:
        predictions (list): Predicted class labels.
    """
    # Step 1: Load the CSV file
    print("Loading input data from CSV...")
    data = pd.read_csv(csv_file)
    X = torch.tensor(data.values, dtype=torch.float32)  # Convert to tensor

    # Step 2: Prepare the Model
    print("Loading the model...")
    input_dim = X.shape[1]  # Number of features from CSV
    output_dim = 4  # Number of classes (update if necessary)

    model = AdvancedNNModel(input_dim, output_dim)
    model.load_state_dict(torch.load(pth_file))
    model.eval()

    # Step 3: Predict
    print("Running predictions...")
    with torch.no_grad():
        outputs = model(X)
        _, predictions = torch.max(outputs, 1)  # Get predicted class

    # Return predictions as a list
    return predictions.tolist()