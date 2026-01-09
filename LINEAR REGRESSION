 import numpy as np
 import matplotlib.pyplot as plt
 from sklearn.linear_model import LinearRegression
 from sklearn.metrics import mean_absolute_error, mean_squared_error

 X = np.array([[1],[2],[3],[4],[5]])
 y = np.array([20,40,50,65,80])

 model = LinearRegression()
 model.fit(X,y)

y_pred = model.predict(X)

 mae = mean_absolute_error(y,y_pred)
 mse = mean_squared_error(y,y_pred)
 rmse = np.sqrt(mse)

 print("Mean Absoulte Error (MAE):",mae)
 print("Mean Squared Error (MSE):",mse)
 print("Root Mean Squared Error (RMSE):",rmse)

import numpy as np
 import matplotlib.pyplot as plt
 from sklearn.linear_model import LinearRegression
 from sklearn.metrics import mean_absolute_error, mean_squared_error

 from google.colab import files
 uploaded = files.upload()

 import pandas as pd
 df = pd.read_csv('Daily_matchdata.csv')

df

# Drop rows with NaN in 'Runs' or 'Dream11_ Points'
df_cleaned = df.dropna(subset=['Runs', 'Dream11_ Points'])
model = LinearRegression()
model.fit(df_cleaned[['Runs']], df_cleaned['Dream11_ Points'])

y_pred = model.predict(df_cleaned[['Runs']])

mae = mean_absolute_error(df_cleaned['Dream11_ Points'], y_pred)
mse = mean_squared_error(df_cleaned['Dream11_ Points'], y_pred)
rmse = np.sqrt(mse)

 print("Mean Absoulte Error (MAE):",mae)
 print("Mean Squared Error (MSE):",mse)
 print("Root Mean Squared Error (RMSE):",rmse)
