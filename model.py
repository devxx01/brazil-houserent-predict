import pandas as pd
import numpy as np

df=pd.read_csv('houses_to_rent_v2.csv')

## Import ML libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import pickle

df1 = df.drop(["hoa (R$)","property tax (R$)","fire insurance (R$)","total (R$)"], axis=1)
# print(df1.head())

#perubahan datatype animal ke bool
boleh=(df1['animal'][0])
tidak=(df1['animal'][4])

for i in range(10692):
  if df1['animal'][i]==boleh:
    df1['animal'][i]=int(1)
  elif df1['animal'][i]==tidak:
    df1['animal'][i]=int(0)

strip=df1['floor'][5]
for i in range(10692):
  if df1['floor'][i]==strip:
    df1['floor'][i]=int(1)

furnish=(df1['furniture'][0])
unfurnish=(df1['furniture'][1])

for i in range(10692):
  if df1['furniture'][i]==furnish:
    df1['furniture'][i]=int(1)
  elif df1['furniture'][i]==unfurnish:
    df1['furniture'][i]=int(0)

df1["floor"]=df1["floor"].astype(int)
df1["city"]=df1["city"].astype(str)

#remove outlier from rent amount
q_low = df1["rent amount (R$)"].quantile(0.01)
q_hi  = df1["rent amount (R$)"].quantile(0.8)

#remove outlier from rent amount
f_low = df1["floor"].quantile(0.01)
f_hi  = df1["floor"].quantile(0.9)

#remove outlier from area
a_hi  = df1["area"].quantile(0.75)

df1_filtered = df1[(df1["rent amount (R$)"] < q_hi) & (df1["rent amount (R$)"] > q_low) & (df1["floor"] < f_hi) & (df1["floor"] > f_low) & (df1["area"] < a_hi)]
# print(df1_filtered.describe())

predictors = ['city', 'area', 'rooms', 'bathroom', 'parking spaces', 'floor', 'animal', 'furniture']
label = 'rent amount (R$)'

#Label Encoder
le = dict()
for column in df1_filtered.columns:
  if df1_filtered[column].dtype == np.object:
    le[column] = LabelEncoder()
    df1_filtered[column] = le[column].fit_transform(df1_filtered[column])

df_train, df_test, y_train, y_test = train_test_split(df1_filtered[predictors], df1_filtered[label], train_size=0.7, random_state=42)
# print(df_train.shape)
# print(df_test.shape) 

#MEMBUAT MODEL TRAIN PARAMETER DEFAULT
def evaluatetrain(model, X=df_train, y=y_train):
  predictions = model.predict(X=df_train)
  errors = abs(predictions - y_train)
  mape = 100 * np.mean(errors / y_train)
  accuracy = 100 - mape
#   print('Model Performance Training')
#   print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
#   print('Accuracy = {:0.2f}%.'.format(accuracy))

  return accuracy

base_model = RandomForestRegressor(random_state=42)
base_model.fit(X=df_train, y=y_train)
base_accuracytrain = evaluatetrain(base_model, X=df_train, y=y_train)

def evaluatetest(model, X=df_test, y=y_test):
  predictionstest = model.predict(X=df_test)
  errorstest = abs(predictionstest - y_test)
  mapetest = 100 * np.mean(errorstest / y_test)
  accuracytest = 100 - mapetest
#   print('Model Performance Testing')
#   print('Average Error: {:0.4f} degrees.'.format(np.mean(errorstest)))
#   print('Accuracy = {:0.2f}%.'.format(accuracytest))

  return accuracytest

base_model = RandomForestRegressor(random_state=42)
base_model.fit(X=df_test, y=y_test)
base_accuracytest = evaluatetest(base_model, X=df_test, y=y_test)

best_gridtrain = RandomForestRegressor(random_state=42, max_features='sqrt', n_estimators=1000)
best_gridtrain.fit(X=df_train, y=y_train)
grid_accuracytrain = evaluatetrain(best_gridtrain, X=df_train, y=y_train)

best_gridtest = RandomForestRegressor(random_state=42, max_features=3, n_estimators=1500)
best_gridtest.fit(X=df_test, y=y_test)
grid_accuracytest = evaluatetest(best_gridtest, X=df_test, y=y_test)

y_pred = best_gridtest.predict(df_test)
print(y_pred)

# Saving model to disk
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(best_gridtest, open('model.pkl','wb'))