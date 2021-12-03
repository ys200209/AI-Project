import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
# import sweetviz as sv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split





data = pd.read_csv('./Project/archive/cardio_train.csv', sep = ';', header = None)
data.to_csv('./Project/archive/cardio_train2.csv', index=False, header = None)





