import os
import json
import pandas as pd
import pyshark
import csv
import asyncio
import re
from collections import defaultdict
from urllib.parse import urlparse
from flask import Flask, render_template, request, send_from_directory, jsonify
from tld import get_tld
import joblib
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # For progress bar (optional)
from same import AdvancedNNModel
from same2 import LSTMModel
from nettt import predict_from_netcsv
from model_prediction import predict_new_data

app = Flask(__name__)

class_names = ["safe", "safe", "not safe", "not safe"]

############################################################################
# URL Prediction
def predict_from_csvurl(csv_file, pth_file, class_names, batch_size=32):
    """
    Load a CSV file and a saved PTH model file, then predict the class names.

    Args:
        csv_file (str): Path to the CSV file with input features.
        pth_file (str): Path to the saved PTH file (model weights).
        class_names (list): List of class names (e.g., ['class_1', 'class_2', ...]).
        batch_size (int): Number of samples to process in each batch (default 32).

    Returns:
        predictions (list): List of predicted class names.
    """
    # Step 1: Load the CSV file
    print("Loading input data from CSV...")
    try:
        data = pd.read_csv(csv_file)
        print(data.head())
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{csv_file}' not found.")

    # Check if the CSV file is empty
    if data.empty:
        raise ValueError(f"CSV file '{csv_file}' is empty.")

    X = torch.tensor(data.values, dtype=torch.float32)  # Convert to tensor

    # Step 2: Prepare the Model
    print("Loading the model...")
    try:
        input_dim = X.shape[1]  # Number of features from CSV
        output_dim = len(class_names)  # Number of classes based on provided class names

        model = AdvancedNNModel(input_dim, output_dim)
        model.load_state_dict(torch.load(pth_file))
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading model or weights: {e}")

    # Step 3: Create DataLoader for batch processing
    dataset = TensorDataset(X)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Step 4: Predict in batches
    print("Running predictions...")
    predictions = []
    with torch.no_grad():
        for inputs in tqdm(data_loader, desc="Predicting", leave=False):
            inputs = inputs[0]  # Extract input tensor from the tuple (inputs, labels)
            outputs = model(inputs)
            _, batch_predictions = torch.max(outputs, 1)  # Get predicted class index for each sample
            predictions.extend([class_names[pred] for pred in batch_predictions.int().tolist()])  # Map to class names

    # Return predictions as a list of class names
    return predictions

############################################################################
csv_file_path = r"D:\Cyber-Security-with-deep-learning-main\Cyber-Security-with-deep-learning-main\GUI\cloud_test.csv"
numerical_cols = ['eventVersion']

categorical_cols = [
    'userAgent', 'eventName', 'awsRegion', 'userIdentitytype', 'userIdentityaccountId',
    'userIdentityprincipalId', 'userIdentityarn', 'userIdentityaccessKeyId', 'userIdentityuserName', 'errorCode'
]

# Load the JIT model
loaded_model = torch.jit.load(r"D:\Cyber-Security-with-deep-learning-main\Cyber-Security-with-deep-learning-main\GUI")

# Load the scaler
loaded_scaler = joblib.load(r"D:\Cyber-Security-with-deep-learning-main\Cyber-Security-with-deep-learning-main\GUI\scaler.save")
############################################################################
net_labels = ['BENIGN', 'DoS Hulk', 'FTP-Patator', 'PortScan', 'DDoS',
              'DoS Slowhttptest', 'DoS slowloris', 'Web Attack – XSS', 'Bot',
              'DoS GoldenEye', 'SSH-Patator', 'Web Attack – Brute Force',
              'Infiltration', 'Web Attack – SQL Injection', 'Heartbleed']

# Model parameters (update these with your model specifics)
input_size = 9   # The correct input size used during training
hidden_size = 64  # The correct hidden size used during training
num_classes = 15  # The correct number of classes for the output layer

# Create an instance of the model
model_net = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

# Load the saved model weights
model_net.load_state_dict(torch.load('net_model.pth'))
model_net.eval()  # Set to evaluation mode
############################################################################
# URL Uploading
# Function to process the URL and extract features
def process_single_url(url):
    # 1. URL Length
    url_len = len(url)

    # 2. Process domain (TLD extraction)
    def process_tld(url):
        try:
            res = get_tld(url, as_object=True, fail_silently=False, fix_protocol=True)
            return res.parsed_url.netloc
        except:
            return None

    domain = process_tld(url)

    # 3. Count the number of specific characters in URL
    features = ['@', '?', '-', '=', '.', '#', '%', '+', '$', '!', '*', '"', ',', '//']
    feature_counts = {feature: url.count(feature) for feature in features}

    # 4. Check for abnormal URL pattern (repeating hostname)
    def abnormal_url(url):
        hostname = urlparse(url).hostname
        return 1 if re.search(hostname, url) else 0

    abnormal_url_flag = abnormal_url(url)

    # 5. Check if the URL is using HTTPS
    def httpSecure(url):
        return 1 if urlparse(url).scheme == 'https' else 0

    https_flag = httpSecure(url)

    # 6. Count digits in the URL
    def digit_count(url):
        return sum(1 for char in url if char.isnumeric())

    digit_count_value = digit_count(url)

    # 7. Count letters in the URL
    def letter_count(url):
        return sum(1 for char in url if char.isalpha())

    letter_count_value = letter_count(url)

    # 8. Check if URL is from a shortening service
    def shortening_service(url):
        match = re.search(r'bit\.ly|goo\.gl|t\.co|tinyurl|adf\.ly|url4\.eu|short\.to|qr\.net|1url\.com', url)
        return 1 if match else 0

    shortening_flag = shortening_service(url)

    # 9. Count the number of directories in the URL path
    def no_of_dir(url):
        urldir = urlparse(url).path
        return urldir.count('/')

    dir_count = no_of_dir(url)

    # 10. Check for suspicious words in URL (e.g., 'login', 'paypal')
    def suspicious_words(url):
        match = re.search(r'PayPal|login|signin|bank|account|update|free|service|bonus|ebayisapi|webscr', url)
        return 1 if match else 0

    suspicious_flag = suspicious_words(url)

    # 11. Calculate hostname length
    hostname_length = len(urlparse(url).netloc)

    # 12. Count the number of uppercase letters in the URL
    upper_count = sum(1 for char in url if char.isupper())

    # 13. Count the number of lowercase letters in the URL
    lower_count = sum(1 for char in url if char.islower())

    # 14. Check if the URL has a "www" prefix
    has_www = 1 if 'www.' in url else 0

    # 15. Count number of subdomains (split by '.')
    subdomain_count = len(urlparse(url).hostname.split('.')) - 2 if urlparse(url).hostname else 0

    # 16. Count the number of query parameters
    query_count = len(urlparse(url).query.split('&')) if urlparse(url).query else 0

    # 17. Count the number of fragments in the URL
    fragment_count = 1 if urlparse(url).fragment else 0

    # 18. Check if the URL uses a port number
    has_port = 1 if urlparse(url).port else 0

    # 19. Count the number of slashes in the URL
    slash_count = url.count('/')

    # 20. Check if the URL uses a path
    has_path = 1 if urlparse(url).path else 0

    # 21. Check if the URL contains "http"
    contains_http = 1 if 'http' in url else 0

    # 22. Check if the URL contains a valid top-level domain
    valid_tld = 1 if process_tld(url) else 0

    # 23. Check if the URL contains a valid domain (e.g., example.com)
    has_valid_domain = 1 if domain else 0

    # 24. Check if the URL contains the string "secure"
    contains_secure = 1 if 'secure' in url else 0

    # Create feature vector
    features = [
        url_len, domain, feature_counts, abnormal_url_flag, https_flag, digit_count_value,
        letter_count_value, shortening_flag, dir_count, suspicious_flag, hostname_length, upper_count,
        lower_count, has_www, subdomain_count, query_count, fragment_count, has_port, slash_count, has_path,
        contains_http, valid_tld, has_valid_domain, contains_secure
    ]

    return features

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.method == "POST":
            # Get the URL from the form
            url = request.form["url"]

            # Process the URL and extract features
            features = process_single_url(url)

            # Run the model prediction
            prediction = model_net(features)  # Forward pass

            # Convert prediction to label
            predicted_class = net_labels[torch.argmax(prediction).item()]

            return jsonify({"prediction": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
