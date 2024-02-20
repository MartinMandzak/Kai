"""Kai v2.0.0"""


#imports
import pandas as pd
import nasdaqdatalink as ndl
ndl.ApiConfig.api_key = "cV_5mV1fxDi_RfMox_wQ"
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, datasets, svm
from sklearn.model_selection import train_test_split

#vars   X == features Y == labels
data = ndl.get('FRED/GDP')


#main

print(data.head())
