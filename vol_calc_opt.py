import pandas as pd
import numpy as np

df = pd.read_csv('cleaned_stocks_dataset.csv')
df['IPO_Date'] = pd.to_datetime(df['IPO_Date'])
df['Date'] = pd.to_datetime(df['Date'])

df['Vol'] = np.nan
df['VV'] = np.nan
stonks=df['Symbol'].unique()
for stock in stonks:
    this_stock = df[df['Symbol'] == stock].copy()
    ipo_date = this_stock['IPO_Date'].iloc[0]
    this_stock = this_stock[this_stock['Date'] >= ipo_date].sort_values('Date')

    if this_stock.empty:
        continue

    # Compute rolling 30-day volatility (std dev of price)
    this_stock['Vol'] = this_stock['Close'].rolling(window=30).std(ddof=0)

    # Compute rolling volatility-of-volatility
    this_stock['VV'] = this_stock['Vol'].rolling(window=30).std(ddof=1)

    # Assign back to original df using original indices
    df.loc[this_stock.index, ['Vol', 'VV']] = this_stock[['Vol', 'VV']].values
    df.loc[this_stock.index, ['days_since_ipo']]=(this_stock['Date']-ipo_date).dt.days
df=df.dropna()
df.to_csv('calculated_stocks.csv', index=False)
