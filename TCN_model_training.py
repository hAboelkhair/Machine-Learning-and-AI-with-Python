#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/hammaad2002/Solar-Irradiance-Forecasting/blob/main/Solar_Irradiance_Forecasting_(comparison_between_RNN%2C_GRU%2C_LSTM%2C_TCN).ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import subprocess

# # Install openpyxl using pip
# subprocess.run(['pip', 'install', 'pandas'])
# subprocess.run(['pip', 'install', 'matplotlib'])
# subprocess.run(['pip', 'install', 'scikit-learn'])
# subprocess.run(['pip', 'install', 'python-math'])
# subprocess.run(['pip', 'install', 'pyinstaller'])
# subprocess.run(['pip', 'install', 'tensorflow'])
# subprocess.run(['pip', 'install', 'keras-tcn', '--no-dependencies'])
# subprocess.run(['pip', 'install', 'protobuf'])
# subprocess.run(['pip', 'install', 'openpyxl'])
# subprocess.run(['pip', 'install', 'tqdm'])
# subprocess.run(['pip', 'install', 'expecttest'])


# In[2]:


# #pip install python==3.11.1
# pip install pandas
# pip install matplotlib
# pip install scikit-learn
# pip install python-math
# pip install pyinstaller
# pip install tensorflow
# pip install keras
# pip install keras-tcn --no-dependencies
# pip install protobuf
# pip install openpyxl


# In[3]:


import numpy as np
np.seterr(divide='ignore', invalid='ignore')


# In[4]:


from tcn import TCN
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import math
from tqdm.notebook import tqdm
import random
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
from math import sin, cos, asin, acos, sqrt
from dateutil.tz import *
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import confusion_matrix, classification_report
import csv
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
import numpy as np
import json
import time as t


# In[5]:


# def make_df_nonzero(df):
#     df_no_zero = df.loc[df['GHI(Wm-2)']!= 0 ]
#     assert len(df_no_zero['DHI(Wm-2)'])==len(df_no_zero['GHI(Wm-2)']), "Some DHI 0s where GHI is not 0"
#     print(df_no_zero)
#     return df_no_zero 


# In[6]:


# Setting seeds value for results reproducability
np.random.seed(42)
random.seed(42) 
tf.random.set_seed(42)


# In[7]:


# List of sheet names or indices in the Excel file
#sheet1 is 'lat_21-lon_24'
# 'Sheet2'  is  'lat_21-lon_26'
# 'Sheet3'  is  'lat_21-lon_28'
# 'Sheet4'  is  'lat_21-lon_30'
# 'Sheet5'  is  'lat_21-lon_32'
# 'Sheet6'  is  'lat_21-lon_34'
# 'Sheet7'  is  'lat_21-lon_36'
# 'Sheet8'  is  'lat_21-lon_38'
sheet_names = ['lat_21-lon_24', 'lat_21-lon_26', 'lat_21-lon_28', 'lat_21-lon_30', 'lat_21-lon_32', 'lat_21-lon_34', 'lat_21-lon_36', 'lat_21-lon_38',
               'lat_23-lon_24', 'lat_23-lon_26', 'lat_23-lon_28', 'lat_23-lon_30', 'lat_23-lon_32', 'lat_23-lon_34', 'lat_23-lon_36', 'lat_23-lon_38',
               'lat_25-lon_24', 'lat_25-lon_26', 'lat_25-lon_28', 'lat_25-lon_30', 'lat_25-lon_32', 'lat_25-lon_34', 'lat_25-lon_36', 'lat_25-lon_38',
               'lat_27-lon_24', 'lat_27-lon_26', 'lat_27-lon_28', 'lat_27-lon_30', 'lat_27-lon_32', 'lat_27-lon_34', 'lat_27-lon_36', 'lat_27-lon_38',
               'lat_29-lon_24', 'lat_29-lon_26', 'lat_29-lon_28', 'lat_29-lon_30', 'lat_29-lon_32', 'lat_29-lon_34', 'lat_29-lon_36', 'lat_29-lon_38',
               'lat_31-lon_24', 'lat_31-lon_26', 'lat_31-lon_28', 'lat_31-lon_30', 'lat_31-lon_32', 'lat_31-lon_34', 'lat_31-lon_36', 'lat_31-lon_38',
               'lat_33-lon_24', 'lat_33-lon_26', 'lat_33-lon_28', 'lat_33-lon_30', 'lat_33-lon_32', 'lat_33-lon_34', 'lat_33-lon_36', 'lat_33-lon_38']

# Create an empty list to store the dataframes
dataframes = []

# Read each sheet and append the resulting dataframe to the list
for sheet in sheet_names:
    df = pd.read_excel('All_Data.xlsx', sheet_name=sheet)
    dataframes.append(df)

# Concatenate the dataframes into a single training dataset
All_dataset = pd.concat(dataframes)


# In[8]:


original_data = All_dataset


# In[9]:


original_data.shape


# In[10]:


All_dataset_3 = All_dataset


# In[11]:


import pandas as pd

# Convert 'Year', 'Month', 'Day', and 'Hour' columns to datetime format
All_dataset_3['Datetime'] = pd.to_datetime(All_dataset_3[['Year', 'Month', 'Day', 'Hour']])

# Define the date range for the last 3 years
end_date = All_dataset_3['Datetime'].max()
start_date = end_date - pd.DateOffset(years=3)

# Select data for the last 3 years excluding specific dates
exclude_dates = pd.to_datetime(['2019-12-31 23:00:00'])  # Add other dates if needed
last_3years = All_dataset_3[
    (All_dataset_3['Datetime'] >= start_date) & 
    (All_dataset_3['Datetime'] <= end_date) & 
    (~All_dataset_3['Datetime'].isin(exclude_dates))
]

# Drop the 'Datetime' column if it's not needed for further analysis
last_3years = last_3years.drop(columns=['Datetime'])
last_3years


# In[ ]:





# In[12]:


All_dataset_10 = All_dataset


# In[13]:


import pandas as pd

# ... (your existing code)

# Convert 'Year', 'Month', 'Day', and 'Hour' columns to datetime format
All_dataset_10['Datetime'] = pd.to_datetime(All_dataset_10[['Year', 'Month', 'Day', 'Hour']])

# Define the date range for the first 10 years
start_date = All_dataset_10['Datetime'].min()
end_date = start_date + pd.DateOffset(years=10)

# Select data for the first 10 years excluding specific dates
exclude_dates = pd.to_datetime(['2020-01-01 00:00:00'])  # Add other dates if needed
first_10years = All_dataset_10[
    (All_dataset_10['Datetime'] >= start_date) & 
    (All_dataset_10['Datetime'] <= end_date) & 
    (~All_dataset_10['Datetime'].isin(exclude_dates))
]

# Drop the 'Datetime' column if it's not needed for further analysis
first_10years = first_10years.drop(columns=['Datetime'])
first_10years


# In[ ]:





# In[14]:


All_dataset.shape


# In[15]:


last_3years.shape


# In[16]:


first_10years.shape


# In[17]:


# Define a function to print layer names and activations
def print_layer_activations(model):
    for layer in model.layers:
        print(f"Layer: {layer.name} - Activation: {layer.activation}")


# In[18]:


per_day_readings =  24                 # readings in one hour x total number of hours in a day
days = 1                          #day=7 == week       # this variable defines our window length
window_length = days * per_day_readings    # we want our model to look back at the data of 3 days
horizon = 1                                # and then predict the next 15th minute reading


# In[19]:


df= All_dataset[['Year', 'Month', 'Day', 'Hour', 'Total_column_water_vapour(mm)',
       'Total_cloud_cover', '2metre_temperature(C)', 'clear-sky_GHI(Wm-2)',
       'Wind_Direction', 'Wind_Speed(m/sec)','GHI(Wm-2)']]


# In[20]:


df_first_10years = first_10years[['Year', 'Month', 'Day', 'Hour', 'Total_column_water_vapour(mm)',
       'Total_cloud_cover', '2metre_temperature(C)', 'clear-sky_GHI(Wm-2)',
       'Wind_Direction', 'Wind_Speed(m/sec)','GHI(Wm-2)']]


# In[21]:


scaler = MinMaxScaler()
df_normalized_first_10years = pd.DataFrame(scaler.fit_transform(df_first_10years), columns=df_first_10years.columns)


# In[22]:


#x= x_normalized.values                  # converting dataframe to numpy arrays
#y= y_normalized.values                  # converting dataframe to numpy arrays
#from tqdm.notebook import tqdm

df_df_normalized_first_10years = df_normalized_first_10years.values                  # converting dataframe to numpy arrays

stride = 1
x = []
y = []
for i in range(0, len(df_df_normalized_first_10years) - window_length, stride):
    x.append(df_df_normalized_first_10years[i:window_length+i, : ])    # GHI included but its past values only
    y.append(df_df_normalized_first_10years[window_length+i,-1])       # next GHI value as our target which is to be predicted by the model


# In[23]:


x_train = np.array(x)                            # converting our features to numpy array
y_train = np.array(y)                            # converting our target to numpy array
print(x_train.shape)                             # printing our total feature data's shape
print(y_train.shape)                             # printing our target data's shape


# In[24]:


df_last_3years = last_3years[['Year', 'Month', 'Day', 'Hour', 'Total_column_water_vapour(mm)',
       'Total_cloud_cover', '2metre_temperature(C)', 'clear-sky_GHI(Wm-2)',
       'Wind_Direction', 'Wind_Speed(m/sec)','GHI(Wm-2)']]


# In[25]:


df_last_3years.shape


# In[26]:


scaler = MinMaxScaler()
df_normalized_last_3years = pd.DataFrame(scaler.fit_transform(df_last_3years), columns=df_last_3years.columns)


# In[27]:


df_df_normalized_last_3years = df_normalized_last_3years.values                  # converting dataframe to numpy arrays

stride = 1
x_test = []
y_test = []
for i in range(0, len(df_df_normalized_last_3years) - window_length, stride):
    x_test.append(df_df_normalized_last_3years[i:window_length+i, : ])    # GHI included but its past values only
    y_test.append(df_df_normalized_last_3years[window_length+i,-1])       # next GHI value as our target which is to be predicted by the model
    

x_test = np.array(x_test)                            # converting our features to numpy array
y_test = np.array(y_test)                            # converting our target to numpy array
print(x_test.shape)                             # printing our total feature data's shape
print(y_test.shape)                             # printing our target data's shape


# In[28]:


# Save x_test to a CSV file
df_last_3years.to_csv('df_last_3years.csv', index=False)


# In[29]:


set_epochs = 200
epochs = set_epochs


# In[ ]:





# ## RNN

# In[39]:


# RNN Model  ( paper is using 6 features that's why its parameters is 4609 )


# modelRNN = tf.keras.Sequential([
#     keras.layers.SimpleRNN(256, input_shape=(24, 11), return_sequences=False), 
#     keras.layers.Dense(1, activation="linear")
# ])
# # Define the learning rate
# learning_rate = 0.0001

# # Compile the model with the optimizer and learning rate
# modelRNN.compile(loss='mae',
#               optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#               metrics=[tf.keras.metrics.RootMeanSquaredError()])
# modelRNN.summary()


# In[40]:


# time1 = t.time()
# print('Train...')
# epochs = set_epochs
# historyRNN = modelRNN.fit(x_train, y_train,
#                     epochs=epochs,                                                          
#                     verbose=1, batch_size=8192)
# timeRNN = t.time() - time1
# print(f"Total time to train {epochs} of RNN model is {timeRNN}")


# # In[51]:


# # Specify the file path
# file_path = 'training_history_RNN.json'

# # Save the history to a JSON file
# with open(file_path, 'w') as file:
#     json.dump(historyRNN.history, file)


# In[41]:


# # RNN prediction
# t1 = t.time()
# predictionRNN = modelRNN.predict(x_test)
# inferRNN = t.time() - t1


# # In[42]:


# # Assuming predictionRNN is a 1D array or a single-column matrix
# predictions_df = pd.DataFrame({'Predictions': predictionRNN.flatten()})

# # Save the predictions to a CSV file
# predictions_df.to_csv('predictions_RNN.csv', index=False)


# # In[43]:


# # Calculate R2 Score
# r2_RNN = r2_score(y_test, predictionRNN.reshape(predictionRNN.shape[0]))
# print("R2 Score:", r2_RNN)

# # Calculate MSE
# mse_RNN = np.mean(np.square(y_test - predictionRNN.reshape(predictionRNN.shape[0])))
# print("MSE:", mse_RNN)

# # Calculate MAE
# mae_RNN = mean_absolute_error(y_test, predictionRNN.reshape(predictionRNN.shape[0]))
# print("MAE:", mae_RNN)

# # Calculate RMSE
# rmse_RNN = np.sqrt(mean_squared_error(y_test, predictionRNN.reshape(predictionRNN.shape[0])))
# print("RMSE:", rmse_RNN)

# Calculate MAPE
# mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
# mape_RNN = np.mean(np.abs((y_test - predictionRNN.reshape(predictionRNN.shape[0])) / y_test)) * 100
# print("MAPE:", mape_RNN)


# In[52]:


# # Specify the file name
# csv_file_name = 'output_RNN.csv'

# # Open the CSV file in write mode ('w', newline='') to ensure proper line endings
# with open(csv_file_name, 'w', newline='') as csvfile:
#     # Create a CSV writer object
#     csv_writer = csv.writer(csvfile)

#     # Write the header row
#     csv_writer.writerow(['Metric', 'Value'])

#     # Write the data rows
#     csv_writer.writerow(['R2 Score', r2_RNN])
#     csv_writer.writerow(['MSE', mse_RNN])
#     csv_writer.writerow(['MAE', mae_RNN])
#     csv_writer.writerow(['RMSE', rmse_RNN])

# # Inform the user that the data has been saved
# print(f"Output has been saved to '{csv_file_name}'")


# In[53]:


# modelRNN.save('RNN_model_1dayV1.h5')


# ## LSTM

# In[54]:


# # LSTM Model
# modelLSTM = tf.keras.Sequential([
#     keras.layers.LSTM(64, input_shape=(24, 11), return_sequences=True),
#     keras.layers.LSTM(128, return_sequences=False),
#     keras.layers.Dense(1, activation="linear")
# ])
# # Define the learning rate
# learning_rate = 0.001

# # Compile the model with the optimizer and learning rate
# modelLSTM.compile(loss='mae',
#               optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#               metrics=[tf.keras.metrics.RootMeanSquaredError()])

# modelLSTM.summary()


# # In[55]:


# time1 = t.time()
# print('Train...')
# epochs = set_epochs
# historyLSTM = modelLSTM.fit(x_train, y_train,
#                     epochs=epochs,                                                          
#                     verbose=1, batch_size=8192)
# timeLSTM = t.time() - time1
# print(f"Total time to train {epochs} of LSTM model is {timeLSTM}")


# # In[56]:


# # Specify the file path
# file_path = 'training_history_LSTM.json'

# # Save the history to a JSON file
# with open(file_path, 'w') as file:
#     json.dump(historyLSTM.history, file)
    
    


# # In[57]:


# modelLSTM.save('LSTM_model_1dayV1.h5')


# # In[58]:


# # LSTM prediction
# t1 = t.time()
# predictionLSTM = modelLSTM.predict(x_test)
# inferLSTM = t.time() - t1


# # In[59]:


# # Assuming predictionRNN is a 1D array or a single-column matrix
# predictions_df = pd.DataFrame({'Predictions': predictionLSTM.flatten()})

# # Save the predictions to a CSV file
# predictions_df.to_csv('predictions_LSTM.csv', index=False)


# # In[60]:


# # Calculate R2 Score
# r2_LSTM = r2_score(y_test, predictionLSTM.reshape(predictionLSTM.shape[0]))
# print("R2 Score:", r2_LSTM)

# # Calculate MSE
# mse_LSTM = np.mean(np.square(y_test - predictionLSTM.reshape(predictionLSTM.shape[0])))
# print("MSE:", mse_LSTM)

# # Calculate MAE
# mae_LSTM = mean_absolute_error(y_test, predictionLSTM.reshape(predictionLSTM.shape[0]))
# print("MAE:", mae_LSTM)

# # Calculate RMSE
# rmse_LSTM = np.sqrt(mean_squared_error(y_test, predictionLSTM.reshape(predictionLSTM.shape[0])))
# print("RMSE:", rmse_LSTM)

# # Calculate MAPE
# # mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
# # mape_LSTM = np.mean(np.abs((y_test - predictionLSTM.reshape(predictionLSTM.shape[0])) / y_test)) * 100
# # print("MAPE:", mape_LSTM)



# # In[61]:


# # Specify the file name
# csv_file_name = 'output_LSTM.csv'

# # Open the CSV file in write mode ('w', newline='') to ensure proper line endings
# with open(csv_file_name, 'w', newline='') as csvfile:
#     # Create a CSV writer object
#     csv_writer = csv.writer(csvfile)

#     # Write the header row
#     csv_writer.writerow(['Metric', 'Value'])

#     # Write the data rows
#     csv_writer.writerow(['R2 Score', r2_LSTM])
#     csv_writer.writerow(['MSE', mse_LSTM])
#     csv_writer.writerow(['MAE', mae_LSTM])
#     csv_writer.writerow(['RMSE', rmse_LSTM])

# # Inform the user that the data has been saved
# print(f"Output has been saved to '{csv_file_name}'")

# inferLSTM_df = pd.DataFrame({'Predictions': inferLSTM.flatten()})

# # Save the predictions to a CSV file
# inferLSTM_df.to_csv('inferLSTM_df.csv', index=False)

# In[ ]:





# # GRU 

# In[62]:


# # GRU Model
# modelGRU = tf.keras.Sequential([
#     keras.layers.GRU(64, input_shape=(24, 11), return_sequences=True),
#     keras.layers.GRU(128, return_sequences=False),
#     keras.layers.Dense(1, activation="linear")
# ])
# # Define the learning rate
# learning_rate = 0.001

# # Compile the model with the optimizer and learning rate
# modelGRU.compile(loss='mae',
#               optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#               metrics=[tf.keras.metrics.RootMeanSquaredError()])

# modelGRU.summary()


# # In[63]:


# # Print layer names and activations
# print_layer_activations(modelGRU)


# # In[64]:


# time1 = t.time()
# print('Train...')
# epochs = set_epochs
# historyGRU = modelGRU.fit(x_train, y_train,
#                     epochs=epochs,                                                          
#                     verbose=1, batch_size=8192)
# timeGRU = t.time() - time1
# print(f"Total time to train {epochs} of GRU model is {timeGRU}")


# # In[65]:


# # Specify the file path
# file_path = 'training_history_GRU.json'

# # Save the history to a JSON file
# with open(file_path, 'w') as file:
#     json.dump(historyGRU.history, file)
    
    


# # In[66]:


# modelGRU.save('GRU_model_1dayV1.h5')


# # In[67]:


# # LSTM prediction
# t1 = t.time()
# predictionGRU = modelGRU.predict(x_test)
# inferGRU = t.time() - t1


# # In[68]:


# # Assuming predictionRNN is a 1D array or a single-column matrix
# predictions_df = pd.DataFrame({'Predictions': predictionGRU.flatten()})

# # Save the predictions to a CSV file
# predictions_df.to_csv('predictions_GRU.csv', index=False)


# # In[69]:


# # Calculate R2 Score
# r2_GRU = r2_score(y_test, predictionGRU.reshape(predictionGRU.shape[0]))
# print("R2 Score:", r2_GRU)

# # Calculate MSE
# mse_GRU = np.mean(np.square(y_test - predictionGRU.reshape(predictionGRU.shape[0])))
# print("MSE:", mse_GRU)

# # Calculate MAE
# mae_GRU = mean_absolute_error(y_test, predictionGRU.reshape(predictionGRU.shape[0]))
# print("MAE:", mae_GRU)

# # Calculate RMSE
# rmse_GRU = np.sqrt(mean_squared_error(y_test, predictionGRU.reshape(predictionGRU.shape[0])))
# print("RMSE:", rmse_GRU)

# # Calculate MAPE
# # mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
# # mape_GRU = np.mean(np.abs((y_test - predictionGRU.reshape(predictionGRU.shape[0])) / y_test)) * 100
# # print("mape", mape)
# # print("MAPE:", mape_GRU)



# # In[71]:


# # Specify the file name
# csv_file_name = 'output_GRU.csv'

# # Open the CSV file in write mode ('w', newline='') to ensure proper line endings
# with open(csv_file_name, 'w', newline='') as csvfile:
#     # Create a CSV writer object
#     csv_writer = csv.writer(csvfile)

#     # Write the header row
#     csv_writer.writerow(['Metric', 'Value'])

#     # Write the data rows
#     csv_writer.writerow(['R2 Score', r2_GRU])
#     csv_writer.writerow(['MSE', mse_GRU])
#     csv_writer.writerow(['MAE', mae_GRU])
#     csv_writer.writerow(['RMSE', rmse_GRU])

# # Inform the user that the data has been saved
# print(f"Output has been saved to '{csv_file_name}'")

# inferGRU_df = pd.DataFrame({'Predictions': inferGRU.flatten()})

# # Save the predictions to a CSV file
# inferGRU_df.to_csv('inferGRU_df.csv', index=False)

# # In[ ]:





# ## TCN

# In[72]:


# TCN Model
modelTCN   = keras.models.Sequential([
              TCN(input_shape=(24, 11), 
              kernel_size=15,
              nb_filters=15,
              dilations=[1, 2, 4, 8],
              padding='causal',
              activation='relu',
              return_sequences=False,
              nb_stacks=2,
              use_skip_connections=False,
              use_batch_norm=True
              ),
              keras.layers.Dense(1, activation="linear")])

# Define the learning rate
learning_rate = 0.001

# Compile the model with the optimizer and learning rate
modelTCN.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, weight_decay = 0.001),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

modelTCN.summary()


# In[73]:


time1 = t.time()
print('Train...')
epochs = set_epochs
historyTCN = modelTCN.fit(x_train, y_train,
                    epochs=epochs,
                    verbose=1, batch_size=8192)
timeTCN = t.time() - time1
print(f"Total time to train {epochs} of TCN model is {timeTCN}")


# In[74]:


# Specify the file path
file_path = 'training_history_TCN.json'

# Save the history to a JSON file
with open(file_path, 'w') as file:
    json.dump(historyTCN.history, file)
    
    


# In[75]:


modelTCN.save('TCN_model_1dayV1.h5')


# In[76]:


# LSTM prediction
t1 = t.time()
predictionTCN = modelTCN.predict(x_test)
inferTCN = t.time() - t1


# In[77]:


# Assuming predictionRNN is a 1D array or a single-column matrix
predictions_df = pd.DataFrame({'Predictions': predictionTCN.flatten()})

# Save the predictions to a CSV file
predictions_df.to_csv('predictions_TCN.csv', index=False)


# In[79]:


# Calculate R2 Score
r2_TCN = r2_score(y_test, predictionTCN.reshape(predictionTCN.shape[0]))
print("R2 Score:", r2_TCN)

# Calculate MSE
mse_TCN = np.mean(np.square(y_test - predictionTCN.reshape(predictionTCN.shape[0])))
print("MSE:", mse_TCN)

# Calculate MAE
mae_TCN = mean_absolute_error(y_test, predictionTCN.reshape(predictionTCN.shape[0]))
print("MAE:", mae_TCN)

# Calculate RMSE
rmse_TCN = np.sqrt(mean_squared_error(y_test, predictionTCN.reshape(predictionTCN.shape[0])))
print("RMSE:", rmse_TCN)

# Calculate MAPE
# mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
# mape_TCN = np.mean(np.abs((y_test - predictionTCN.reshape(predictionTCN.shape[0])) / y_test)) * 100
# print("MAPE:", mape_TCN)



# In[80]:


# Specify the file name
csv_file_name = 'output_TCN.csv'

# Open the CSV file in write mode ('w', newline='') to ensure proper line endings
with open(csv_file_name, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)

    # Write the header row
    csv_writer.writerow(['Metric', 'Value'])

    # Write the data rows
    csv_writer.writerow(['R2 Score', r2_TCN])
    csv_writer.writerow(['MSE', mse_TCN])
    csv_writer.writerow(['MAE', mae_TCN])
    csv_writer.writerow(['RMSE', rmse_TCN])

# Inform the user that the data has been saved
print(f"Output has been saved to '{csv_file_name}'")


inferTCN_df = pd.DataFrame({'Predictions': inferTCN.flatten()})

# Save the predictions to a CSV file
inferTCN_df.to_csv('inferTCN_df.csv', index=False)


# In[ ]:





# ## Results

# In[81]:


# # Your existing print statements
# header = ["Model", "R2 Square", "MSE", "MAE", "RMSE"]
# data = [
#     ["LSTM", r2_LSTM, mse_LSTM, mae_LSTM, rmse_LSTM],
#     ["GRU", r2_GRU, mse_GRU, mae_GRU, rmse_GRU],
#     ["RNN", r2_RNN, mse_RNN, mae_RNN, rmse_RNN],
#     ["TCN", r2_TCN, mse_TCN, mae_TCN, rmse_TCN]
# ]

# # Write to CSV file
# csv_filename = "model_metrics.csv"
# with open(csv_filename, "w", newline="") as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(header)
#     writer.writerows(data)

# # Print confirmation message
# print(f"Metrics saved to {csv_filename}")


# In[114]:


# print(f"          R2 Square  /     MSE     /    MAE     /   RMSE     ")
# print(f"LSTM   :  {r2_LSTM:.7f}     {mse_LSTM:.7f}    {mae_LSTM:.7f}    {rmse_LSTM:.7f} ")
# print(f"GRU    :  {r2_GRU:.7f}     {mse_GRU:.7f}    {mae_GRU:.7f}    {rmse_GRU:.7f}  ")
# print(f"RNN    :  {r2_RNN:.7f}     {mse_RNN:.7f}    {mae_RNN:.7f}    {rmse_RNN:.7f}  ")
# print(f"TCN    :  {r2_TCN:.7f}    {mse_TCN:.7f}    {mae_TCN:.7f}    {rmse_TCN:.7f}   ")


# In[105]:


# fig, ax = plt.subplots()

# ax.scatter(y_test, predictionRNN, label='RNN Predictions')

# ax.scatter(y_test, predictionLSTM, label='LSTM Predictions')

# ax.scatter(y_test, predictionGRU, label='GRU Predictions')

# # ax.scatter(y_test, predictionTCN, label='TCN Predictions')

# ax.plot(y_test, y_test, 'r', label='Perfect Predictions')

# # Add labels and legend
# ax.set_xlabel('True Values')
# ax.set_ylabel('Predictions')
# ax.legend()

# # Save the figure as an image file (e.g., PNG)
# plt.savefig('predictions_plot.png')
# # Show the plot
# plt.show()


# In[ ]:





# In[91]:


# fig, ax = plt.subplots()

# losses_RNN = pd.DataFrame(historyRNN.history)
# losses_RNN.plot(ax=ax, label='RNN Losses')

# losses_LSTM = pd.DataFrame(historyLSTM.history)
# losses_LSTM.plot(ax=ax, label='LSTM Losses')

# losses_GRU = pd.DataFrame(historyGRU.history)
# losses_GRU.plot(ax=ax, label='GRU Losses')

# # losses_TCN = pd.DataFrame(historyTCN.history)
# # losses_TCN.plot(ax=ax, label='TCN Losses')

# # Add labels and legend
# ax.set_xlabel('Epochs')
# ax.set_ylabel('Loss')
# ax.legend()

# plt.savefig('Losses_plot.png')

# # Show the plot
# plt.show()


# In[97]:


# # Create subplots for each model
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
# fig.suptitle('Loss Comparison for Different Models', fontsize=16)

# # Flatten the 2x2 subplot grid to access each subplot individually
# axes = axes.flatten()

# # Plot and set titles for each model
# losses_RNN = pd.DataFrame(historyRNN.history)
# losses_RNN.plot(ax=axes[0], label='RNN Losses')
# axes[0].set_title('RNN')

# losses_LSTM = pd.DataFrame(historyLSTM.history)
# losses_LSTM.plot(ax=axes[1], label='LSTM Losses')
# axes[1].set_title('LSTM')

# losses_GRU = pd.DataFrame(historyGRU.history)
# losses_GRU.plot(ax=axes[2], label='GRU Losses')
# axes[2].set_title('GRU')

# losses_TCN = pd.DataFrame(historyTCN.history)
# losses_TCN.plot(ax=axes[3], label='TCN Losses')
# axes[3].set_title('TCN')

# # Add common labels and legend
# for ax in axes:
#     ax.set_xlabel('Epochs')
#     ax.set_ylabel('Loss')
#     ax.legend()

# # Adjust layout
# plt.tight_layout(rect=[0, 0, 1, 0.96])

# # Save each subplot as a separate figure
# for i, ax in enumerate(axes):
#     plt.figure()  # create a new figure
#     ax.get_figure().savefig(f'model_{i+1}_loss_plot.png')

# # Show the original plot
# plt.show()


# In[ ]:





# # **Loss** 

# In[99]:


# Extract the history of loss and metric result from the history object

# # #RNN
# loss_history_rnn = historyRNN.history['loss']
# r2_history_rnn = historyRNN.history['root_mean_squared_error']

# #GRU
# loss_history_gru = historyGRU.history['loss']
# r2_history_gru = historyGRU.history['root_mean_squared_error']

# #LSTM
# loss_history_lstm = historyLSTM.history['loss']
# r2_history_lstm = historyLSTM.history['root_mean_squared_error']

# # #TCN
# loss_history_tcn = historyTCN.history['loss']
# r2_history_tcn = historyTCN.history['root_mean_squared_error']


# # RNN loss and Metric Plot

# In[100]:


# # Plot the history of loss of RNN
# plt.plot(loss_history_rnn)
# plt.title('Loss history of RNN model')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.savefig('loss_history_rnn.png')  # Save the plot as an image file
# plt.show()

# # Plot the history of metric result of RNN
# plt.plot(r2_history_rnn)
# plt.title('Metric result history RNN')
# plt.xlabel('Epoch')
# plt.ylabel('Metric result')
# plt.savefig('metric_result_rnn.png')  # Save the plot as an image file
# plt.show()


# # GRU loss and Metric Plot

# In[101]:


# # Plot the history of loss of GRU
# plt.plot(loss_history_gru)
# plt.title('Loss history of GRU model')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.savefig('loss_history_gru.png')  # Save the plot as an image file
# plt.show()

# # Plot the history of metric result of GRU
# plt.plot(r2_history_gru)
# plt.title('Metric result history GRU')
# plt.xlabel('Epoch')
# plt.ylabel('Metric result')
# plt.savefig('metric_result_gru.png')  # Save the plot as an image file
# plt.show()


# # LSTM loss and Metric Plot

# In[102]:


# # Plot the history of loss of LSTM
# plt.plot(loss_history_lstm)
# plt.title('Loss history of LSTM model')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.savefig('loss_history_lstm.png')  # Save the plot as an image file
# plt.show()

# # Plot the history of metric result of LSTM
# plt.plot(r2_history_lstm)
# plt.title('Metric result history LSTM')
# plt.xlabel('Epoch')
# plt.ylabel('Metric result')
# plt.savefig('metric_result_lstm.png')  # Save the plot as an image file
# plt.show()


# ## TCN loss and Metric Plot

# In[103]:


# # Plot the history of loss of TCN
# plt.plot(loss_history_tcn)
# plt.title('Loss history of TCN model')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.savefig('loss_history_tcn.png')  # Save the plot as an image file
# plt.show()

# # Plot the history of metric result of TCN
# plt.plot(r2_history_tcn)
# plt.title('Metric result history TCN')
# plt.xlabel('Epoch')
# plt.ylabel('Metric result')
# plt.savefig('metric_result_tcn.png')  # Save the plot as an image file
# plt.show()


# In[117]:


# testing_years = 'Last 3 years'

# # Making the figure size bigger
# plt.figure(figsize=(12, 8))

# # RNN's prediction plot
# plt.subplot(2,2,1)
# plt.plot(predictionRNN)
# plt.plot(y_test)
# plt.title(f'{testing_days} day window results of RNN')
# plt.legend(['predicted', 'actual'])
# plt.savefig('rnn_prediction_plot.png')  # Save the subplot as an image file
# plt.clf()  # Clear the current figure to avoid overlapping

# # GRU's prediction plot
# plt.subplot(2,2,2)
# plt.plot(predictionGRU)
# plt.plot(y_test)
# plt.title(f'{testing_days} day window results of GRU')
# plt.legend(['predicted', 'actual'])
# plt.savefig('gru_prediction_plot.png')  # Save the subplot as an image file
# plt.clf()  # Clear the current figure to avoid overlapping

# # LSTM's prediction plot
# plt.subplot(2,2,3)
# plt.plot(predictionLSTM)
# plt.plot(y_test)
# plt.title(f'{testing_days} day window results of LSTM')
# plt.legend(['predicted', 'actual'])
# plt.savefig('lstm_prediction_plot.png')  # Save the subplot as an image file
# plt.clf()  # Clear the current figure to avoid overlapping

# # TCN's prediction plot
# plt.subplot(2,2,4)
# plt.plot(predictionTCN)
# plt.plot(y_test)
# plt.title(f'{testing_days} day window results of TCN')
# plt.legend(['predicted', 'actual'])
# plt.savefig('tcn_prediction_plot.png')  # Save the subplot as an image file
# plt.clf()  # Clear the current figure to avoid overlapping


# In[118]:


# testing_years = 'Last 3 years'

# # Making the figure size bigger
# plt.figure(figsize=(12, 8))

# # RNN's prediction plot
# plt.subplot(2,2,1)
# plt.plot(predictionRNN)
# plt.plot(y_test)
# plt.title(f'{testing_years} day window results of RNN')
# plt.legend(['predicted', 'actual'])
# plt.savefig('rnn_prediction_plot.png')  # Save the subplot as an image file
# plt.clf()  # Clear the current figure to avoid overlapping

# # GRU's prediction plot
# plt.subplot(2,2,2)
# plt.plot(predictionGRU)
# plt.plot(y_test)
# plt.title(f'{testing_years} day window results of GRU')
# plt.legend(['predicted', 'actual'])
# plt.savefig('gru_prediction_plot.png')  # Save the subplot as an image file
# plt.clf()  # Clear the current figure to avoid overlapping

# # LSTM's prediction plot
# plt.subplot(2,2,3)
# plt.plot(predictionLSTM)
# plt.plot(y_test)
# plt.title(f'{testing_years} day window results of LSTM')
# plt.legend(['predicted', 'actual'])
# plt.savefig('LSTM_prediction_plot.png')  # Save the subplot as an image file
# plt.clf()  # Clear the current figure to avoid overlapping


# # In[113]:


# # Making the figure size bigger
# plt.figure(figsize=(12, 8))

# # RNN's prediction plot
# plt.subplot(2,2,1)
# plt.plot(predictionRNN)
# plt.plot(y_test)
# plt.title(f'{testing_years} window results of RNN')
# plt.legend(['predicted', 'actual'])

# # GRU's prediction plot
# plt.subplot(2,2,2)
# plt.plot(predictionGRU)
# plt.plot(y_test)
# plt.title(f'{testing_years} window results of GRU')
# plt.legend(['predicted', 'actual'])

# # LSTM's prediction plot
# plt.subplot(2,2,3)
# plt.plot(predictionLSTM)
# plt.plot(y_test)
# plt.title(f'{testing_years} window results of LSTM')
# plt.legend(['predicted', 'actual'])


# # TCN's prediction plot
# plt.subplot(2,2,4)
# plt.plot(predictionTCN)
# plt.plot(y_test)
# plt.title(f'{testing_years} window results of TCN')
# plt.legend(['predicted', 'actual'])


# # In[109]:


# models = ['RNN', 'GRU', 'LSTM','TCN']
# timeTrain = [timeRNN, timeGRU, timeLSTM, timeTCN]
# timeInfer = [inferRNN, inferGRU, inferLSTM, inferTCN]

# plt.bar(models, timeTrain)
# plt.xlabel('Model')  
# plt.ylabel('Training Time (seconds)')
# plt.title('Model Training Time')
# plt.savefig('trainingTimemodels.png')
# plt.show()


# # In[110]:


# plt.bar(models, timeInfer)
# plt.xlabel('Model')  
# plt.ylabel('Inference Time (seconds)')
# plt.title('Model Inference Time')
# plt.savefig('InferingImg.png')
# plt.show()


# In[ ]:





# In[ ]:




