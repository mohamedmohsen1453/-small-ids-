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

class AdvancedNNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AdvancedNNModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)   # First layer (increase the size for more capacity)
        self.bn1 = nn.BatchNorm1d(128)         # Batch Normalization
        self.dropout1 = nn.Dropout(0.4)        # Dropout regularization

        self.fc2 = nn.Linear(128, 256)         # Second layer (larger hidden layer)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, 256)         # Third layer (reduced again for better capacity control)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.4)

        self.fc4 = nn.Linear(256, output_dim)  # Output layer for 4 classes


    def forward(self, x):
        # Pass through the layers with batch normalization, ReLU activations, and dropout

        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)


        x = self.fc4(x)  # No activation for logits in multi-class classification
        return x