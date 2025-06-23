import pandas as pd
from sklearn.metrics import classification_report, mean_squared_error, confusion_matrix, f1_score, accuracy_score, \
    precision_score, recall_score, auc, roc_curve

from tld import get_tld, is_tld
from sklearn.model_selection import train_test_split
import random
import math
from collections import Counter
import time
from bs4 import BeautifulSoup
import urllib
from urllib.parse import urlparse, parse_qs
from datasets import load_dataset
from datasets import Dataset
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
from collections import Counter
import re
from urllib.parse import urlparse
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class StrongNNBinaryClassification(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(StrongNNBinaryClassification, self).__init__()

        # Define a deeper architecture
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 1)  # Single output neuron for binary classification
        self.dropout = nn.Dropout(0.3)  # Dropout layer

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)  # Output layer (no activation yet)
        return x


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a dummy time dimension
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Output from the last time step
        return out