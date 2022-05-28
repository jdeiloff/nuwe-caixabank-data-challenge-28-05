# -*- coding: utf-8 -*-
"""
Created on Sat May 28 17:52:52 2022

@author: Jony
"""
# Needed libraries for the challenge
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers

# Ingesting data parsing dates to datetime format
df = pd.read_csv('train.csv', index_col='Date',  parse_dates=True, infer_datetime_format=True)

# Analyzing data, checking for null values and dealing with them.
#df.head()

# missing = df.isnull().sum().sum() #798
#df.shape
# total_cell = np.product(df.shape) #45878

df.dropna(axis=0, inplace=True) 

# missing_after_drop = df.isnull().sum().sum() # 0
df.shape
# total_cell_after_drop = np.product(df.shape) #44947

# Select independant variable features

input_var = df[['Open','High','Low','Close','Adj Close','Volume']]
output_var = df[['Target']]


# Splitting train and test sets for validation using sklearn's timeseriessplit

timesplit= TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(input_var):
        X_train, X_test = input_var[:len(train_index)], input_var[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()



# Define F1 Score Metric, taken from old keras source code
def get_f1(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# Define DL model
model = keras.Sequential([
    layers.BatchNormalization(input_shape=(X_train.shape[1],)),
    layers.Dense(400, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(400, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(400, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid'),
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[get_f1],
)


# Train model
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=512,
    epochs=200,
    callbacks=[early_stopping],
)

# Plotting performance
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['get_f1', 'val_get_f1']].plot(title="Accuracy")

# Predict with given data and save results to .csv and .json files.
test_x = pd.read_csv('test_x.csv', index_col='Date',  parse_dates=True, infer_datetime_format=True)
test_x_1 = test_x.drop('test_index', axis=1)
prediction = model.predict(test_x_1)
predictions = pd.DataFrame(prediction, index= test_x.test_index, columns=['Target']).to_csv('predictions.csv')
predictionsjson = pd.DataFrame(predictions, index= test_x.test_index, columns=['Target']).to_json('predictions.json')



