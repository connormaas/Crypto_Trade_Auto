# Display Trade Prices
# NOTE: Certain features have been excluded to prevent exposure of personal 
#       information

# In[2]:

import pandas as pd
from matplotlib import pyplot as plt
import statistics as stat

# In[30]:

products =['REP-USD', 'EOS-USD', 'KNC-USD', 'ZRX-USD', 'ETC-USD',
           'XLM-USD', 'LINK-USD', 'XRP-USD', 'BCH-USD', 'LTC-USD',
           'ETH-USD', 'BTC-USD']
# products =['KNC-USD', 'ETC-USD',
#            'XLM-USD', 'LINK-USD']

# In[31]:

master_df = pd.read_csv('fills (1).csv')

master_df

# In[32]:

prod = 'REP-USD'
df = master_df[master_df['product'] == prod]

# In[36]:

total_profits = pd.Series()
market_changes = pd.Series()
for prod in ['XLM-USD']:
    df = master_df[master_df['product'] == prod]
    percent_profits = []
    last = 'SELL'
    last_buy = None
    sides = df['side'].values
    prices = df['price'].values
    for i in range(len(df)):
        if sides[i] != last:
            if last == 'BUY':
                percent_profits.append((prices[i] - last_buy)/last_buy)
                last = 'SELL'
            else:
                last_buy = prices[i]
                last = 'BUY'
            
    total_profits.set_value(prod,sum(percent_profits))
    market_changes.set_value(prod,(prices[-1]-prices[0])/prices[0])
    
# In[37]:


print('Percent Profit ' + str(round(stat.mean(total_profits)*100,2)) + '%')
print('Percent Profit Buy and Hold ' + str(round(stat.mean(market_changes)*100,2)) + '%')


# In[35]:

plt.style.use('dark_background')

for prod in products:
    df = master_df[master_df['product'] == prod]
    plt.figure(figsize = (20,7))
    df['created at'] = pd.to_datetime(df['created at'])
    buys = df[df['side']=='BUY']
    sells = df[df['side']=='SELL']
    plt.scatter(sells['created at'],sells['price'],label = 'SEll',color = 'red',alpha = .65,s = 20)
    plt.scatter(buys['created at'],buys['price'],label = 'BUY',color = 'green',alpha = .65,s= 20)
    plt.xlim(buys['created at'].values[0],sells['created at'].values[-1])
    plt.xticks(rotation = 45)
    plt.title(prod)
    plt.show()
    plt.close()


# In[ ]: