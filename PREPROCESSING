#Data cleaning
from google.colab import drive
drive.mount('/content/drive')

file_path = '/content/drive/My Drive/MLLAB/Cardetails.csv'

import pandas as pd
df = pd.read_csv(file_path)

df.shape

df.head()

df.info()

df.isnull()

df.isnull().sum()

print(df.isna().sum)

df = df.dropna(axis=0) #indexing start here at 0  #1st row delete krne ho toh

df.isnull().sum()

#Deleting the entire column
df=df.dropna(axis=1)

df.isnull().sum()

# Data cleaning
# Reset the index before dropping rows by index value
df = df.reset_index(drop=True)
df = df.drop(labels=[1, 5, 2], axis=0)

#delete a range of rows - index values 10-20
df=df.drop(labels=range(40,45), axis=0)

#Shortest version of the above
df=df.drop(0)

import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

file_path = '/content/drive/My Drive/MLLAB/pokemon_data.csv'

df = pd.read_csv(file_path)

df.head()

df.shape

df.info()

df.isnull().sum()

df = df.dropna(axis = 0)

df.isnull().sum()

df = df.dropna(axis = 1 )
df.isnull().sum()

data = df.drop(labels=[1, 15, 20], axis=0)

data = data.drop(range(40,50),axis = 0, errors= 'ignore')

data = df.drop(0)
