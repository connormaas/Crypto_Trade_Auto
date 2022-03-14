# Display Positions
# NOTE: - Certain features have been excluded to prevent exposure of personal 
#         information

# In[1]:

import pandas as pd
import os
import time
import cbpro
from tabulate import tabulate

# In[2]:

# In[3]:

def printTable(table):
    print(tabulate(table, headers='keys', tablefmt='fancy_grid',showindex = False))
    print()

# In[4]:

while True:
    try:
        printTable(pd.read_csv('Positions.csv').iloc[:,1:])
        tempdf = pd.DataFrame(pd.read_csv('data_update_times.csv')['Times'])
        tempdf['Buys'] = pd.read_csv('buys.csv')['Buys']
        tempdf['Sells'] = pd.read_csv('sells.csv')['Sells']
        printTable(tempdf.iloc[-5:,:])
    except:
        print()
        print('--- Error ---')
    time.sleep(5)

# In[ ]:

# In[ ]:

# In[ ]: