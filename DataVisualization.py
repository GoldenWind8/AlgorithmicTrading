import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import mplfinance as mpf
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader.data as web

df = pd.read_csv('tsla.csv', parse_dates= True, index_col=0)
#df[['Open', 'Close', 'High']].plot()
#plt.show()

#Creating a moving average column
df['100ma'] = df['Adj Close'].rolling(window =100).mean()

df.dropna(inplace=True)

#Graphing info with matplotlib
ax1 = plt.subplot2grid((6,1), (0,0), 4, 1)
ax2 = plt.subplot2grid((6,1), (5,0), 2, 1, sharex = ax1)  #ShareX links the axes's so zooming one, zooms the other

ax1.plot(df.index , df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])
#plt.show()


#Remoddling Data - We want to resample daily prices to 10 day data
df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

print("Resampled Data")
print(df_ohlc)
print(df_volume)


#Using mplfinance package:
df_ohlc.reset_index(inplace = True)  #Resetting date as not index
print(df_ohlc)

#Print Candlestick chart
mpf.plot(df, type='candle', style='charles',
            title='  ',
            ylabel='  ',
            ylabel_lower='  ',
            figratio=(25,10),
            figscale=1,
            mav=50,
            volume=True
            )

plt.show()
