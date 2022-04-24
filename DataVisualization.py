import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

df = pd.read_csv('tsla.csv', parse_dates= True, index_col=0)
#df[['Open', 'Close', 'High']].plot()
#plt.show()
print(df.head(7))

#Creating a moving average column
df['100ma'] = df['Adj Close'].rolling(window =100).mean()

print(df.head())
print(df.tail())
df.dropna(inplace=True)
print(df.head())

#Graphing info with matplotlib
ax1 = plt.subplot2grid((6,1), (0,0), 4, 1)
ax2 = plt.subplot2grid((6,1), (5,0), 2, 1, sharex = ax1)  #ShareX links the axes's so zooming one, zooms the other

ax1.plot(df.index , df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])
plt.show()