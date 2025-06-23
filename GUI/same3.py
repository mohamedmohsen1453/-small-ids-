import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):

    # Automatically handle input shape mismatches
        if x.dim() == 2:  # If input is (batch_size, input_size)
            x = x.unsqueeze(1)  # Add a sequence dimension

        if x.dim() != 3:
            raise ValueError(f"Expected input to have 3 dimensions, got {x.dim()}D instead")

        out, _ = self.lstm(x)  # Pass through LSTM
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out

# Instantiate the model
