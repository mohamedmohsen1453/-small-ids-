
# Cybersecurity Framework for Analysis

This repository contains a comprehensive framework designed for cybersecurity analysis, focusing on three key areas:

1. **Cloud Analysis**: Analyzing JSON files.
2. **Network Analysis**: Parsing and analyzing PCAP files.
3. **Phishing & Web Analysis**: Inspecting URLs for phishing and other threats.

We have implemented three machine learning models and integrated a web-based GUI for ease of use.

## Features

- **Models**:

  - **Cloud Analysis**: Advanced Neural Network (ANN) for JSON file analysis.
  - **Network Analysis**: Long Short-Term Memory (LSTM) model for PCAP file analysis.
  - **Phishing URL Analysis**: Advanced Neural Network (ANN) for URL classification.

- **Web GUI**:

  - Built using Flask.
  - Allows users to upload files or URLs and view the analysis results.

## Requirements

To run this framework, you need the following libraries and tools installed:

```bash
os
json
pandas
pyshark
csv
asyncio
re
defaultdict (from collections)
urlparse (from urllib.parse)
Flask
joblib
torch
seaborn
matplotlib
sklearn
keras
plotly
```

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/cybersecurity-framework.git
   cd cybersecurity-framework
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:

   ```bash
   python app.py
   ```

## File Structure

```
cybersecurity-framework/
|
├── GUI/
│   ├── app.py                  # Flask application
│   ├── uploads/                # Directory for uploaded files
│   ├── static/                 # Static files (CSS, JS, images)
│   ├── templates/              # HTML templates for the Flask app
│   ├── nettt.py                # Network data prediction functions
│   ├── same.py                 # ANN for URL analysis
│   ├── same2.py                # LSTM for network analysis
│   ├── same3.py                # Additional utility functions
│   ├── model_prediction.py     # Generic model prediction functions
│   ├── advanced_nn_model.pth   # Pre-trained model for cloud analysis
│   ├── net_model.pth           # Pre-trained model for network analysis
│   ├── model_traced.pt         # Traced model for deployment
│   ├── scaler.save             # Saved scaler for data preprocessing
│   ├── cloud_test.csv          # Sample test file for cloud analysis
│   ├── network_test.csv        # Sample test file for network analysis
│   ├── url_test.csv            # Sample test file for URL analysis
├── models/
│   ├── cloud/
│   │   ├── data-cleaning-and-model-training/
│   │   │   ├── cloud_ids copy.ipynb  # Cloud data preprocessing and model training
│   │   │   ├── model_prediction.py   # Prediction logic
│   │   │   ├── model_traced.pt       # Traced model for cloud deployment
│   │   │   ├── scaler.save           # Scaler for preprocessing
│   │   │   ├── test.json             # Sample JSON file for testing
│   ├── network/
│   │   ├── network_ids.ipynb         # Network model training notebook
│   │   ├── network_test.csv          # Sample CSV for network testing
│   │   ├── test.pcap                 # Sample PCAP file for analysis
│   │   ├── trai_model.pth            # Trained model for network analysis
│   ├── url/
│   │   ├── advanced_nn_model.pth     # Pre-trained model for URL analysis
│   │   ├── p.py                      # Helper script for URL analysis
│   │   ├── url_doc.pdf               # Documentation for URL analysis
│   │   ├── url_ids.ipynb             # Notebook for URL analysis
│   │   ├── url_model.ipynb           # Model training for URL analysis
├── documentation/
│   ├── presentation.pptx             # Project presentation
│   ├── detailed_documentation.pdf    # Comprehensive project documentation
├── README.md                         # Documentation (this file)
```

## Usage

1. Open the web application in your browser (default: `http://127.0.0.1:5000`).
2. Select the type of analysis (Cloud, Network, or Phishing).
3. Upload your file or enter the URL for analysis.
4. View the results on the web interface.

## Key Python Libraries Used

- **Data Handling**: `pandas`, `numpy`
- **Visualization**: `seaborn`, `matplotlib`, `plotly`
- **Web Framework**: `Flask`
- **Machine Learning**: `torch`, `keras`
- **Utilities**: `asyncio`, `collections`, `urlparse`

## Models Description

### Cloud Analysis

- Model: Advanced Neural Network (ANN)
- Purpose: Analyze JSON files for cybersecurity insights.

### Network Analysis

- Model: Long Short-Term Memory (LSTM)
- Purpose: Analyze PCAP files for network threat detection.

### Phishing URL Analysis

- Model: Advanced Neural Network (ANN)
- Purpose: Classify URLs to detect phishing attempts.

## Contributors

- Samir Walid
- Ahmed Fawzy
- Omar Hamed
- Ahmed Elgohary
- Ahmed Gamal
- Basant Emad
- Amira Asem
- Fayrouz Eslam


---

Feel free to contribute or raise issues to improve the framework!
