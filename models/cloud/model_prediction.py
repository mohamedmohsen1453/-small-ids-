import pandas as pd
import torch
import joblib

def predict_new_data(csv_file_path, model, scaler, numerical_cols, categorical_cols):
    # Load the new dataset
    new_data = pd.read_csv(csv_file_path)
    
    # Preprocess numerical data (Standardization)
    X_num_new = scaler.transform(new_data[numerical_cols])
    
    # Preprocess categorical data (Factorize)
    X_cat_new = new_data[categorical_cols].apply(lambda x: pd.factorize(x)[0])
    
    # Convert to tensors
    X_num_new_tensor = torch.tensor(X_num_new, dtype=torch.float32)
    X_cat_new_tensor = torch.tensor(X_cat_new.values, dtype=torch.long)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        outputs = model(X_num_new_tensor, X_cat_new_tensor)
        predictions = torch.sigmoid(outputs).round().numpy()
    
    return predictions

# Example usage:
csv_file_path = r"C:\Users\Omar\Desktop\full_deep_pro\Security-with-Deep-learning-master\models\cloud\data-cleaning-and-model-training\test.csv"
numerical_cols = ['eventVersion']

categorical_cols = [
    'userAgent', 'eventName', 'awsRegion', 'userIdentitytype', 'userIdentityaccountId',
    'userIdentityprincipalId', 'userIdentityarn', 'userIdentityaccessKeyId', 'userIdentityuserName', 'errorCode'
]



# Load the JIT model
loaded_model = torch.jit.load("model_traced.pt")

# Load the scaler
loaded_scaler = joblib.load("scaler.save")

# Make predictions
predictions = predict_new_data(csv_file_path, loaded_model, loaded_scaler, numerical_cols, categorical_cols)
print(predictions)


