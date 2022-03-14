# Algorithm Execution File
# NOTE: - Certain features have been excluded to prevent exposure of personal 
#         information

# In[5]:

import pandas as pd 
import time 
import statistics as stat
import datetime
import cbpro
import requests
import yfinance as yf
import numpy as np
import os
from twilio.rest import Clients

crypto_list = ['BTC','XLM','ETH','XRP','EOS','LTC','BCH','ETC','LINK','REP','ZRX','KNC']
not_listed_cryptos = ['XTZ','ALGO','OXT','ATOM']
data_dic = {}
original_data_dic = {}

# In[3]:

def ljust(sarr):
    tempsarr = []
    for i in sarr:
        tempsarr.append(i.ljust(9))
    return tempsarr

def success_rate(lst):
    lst = np.array(list(lst))
    return len(lst[lst>0])/len(lst)
    
def getdata(symbol, daysback, printtime = True):
    st = time.time()
    dta = yf.download(tickers = symbol.upper()+'-USD', interval = '1h', period = str(daysback) + 'd', progress = False)
    if printtime:   
        print('Download Time | '+ str(time.time()-st))
    dta.reset_index(inplace = True)
    dta['Time'] = dta['Date']
    dta.drop(columns = ['Date','Volume'],inplace = True)
    return dta

def makeTimes(unix):
    date = datetime.datetime.utcfromtimestamp(unix)
    return date

def makeRSI(df):
    period = 7
    rsi_list = []
    for i in range(period,len(df)):
        tempdata = df.iloc[i-period:i,:] 
        
        if len(tempdata.loc[tempdata.loc[:,'Gain'],'Change'])==0:
            gains = 0
        else:          
            gains = stat.mean(tempdata.loc[tempdata.loc[:,'Gain'],'Change'])    
        if len(tempdata.loc[tempdata.loc[:,'Gain']==False,'Change'])==0:
            loses = 0
        else:    
            loses = stat.mean(tempdata.loc[tempdata.loc[:,'Gain']==False,'Change'].apply(abs))
        if loses == 0:
            rsi_list.append(100)
        else:    
            rsi_list.append(100 - 100/(1+(gains/loses)))
            
    return [0]*period + rsi_list

def makeRSI(df):
    period = 7
    rsi_list = []
    full_gains = df.loc[:,'Gain']
    full_changes = df.loc[:,'Change']
    for i in range(period,len(df)):
        temp_gains = full_gains[i-period:i] 
        temp_changes = full_changes[i-period:i]
        
        if (~temp_gains).all():
            gains = 0
        else:          
            gains = stat.mean(temp_changes[temp_gains])    
        
        if (temp_gains).all():
            loses = 0
        else:    

            loses = stat.mean(temp_changes[~temp_gains])
        
        if loses == 0:
            rsi_list.append(100)
        else:    
            rsi_list.append(100 - 100/(1+(gains/loses)))
            
    return [0]*period + rsi_list

def makeOBV(df):
    period = 10
    obv_list = []
    for i in range(period,len(df)):
        tempdata = df.iloc[i-period:i,:]
        gains = sum(tempdata[tempdata['Gain']]['Change'])
        loses = sum(tempdata[tempdata['Gain']==False]['Change'])
        obv_list.append(gains+loses)
    return [0]*period + obv_list

def makeSMA(Series,period):
    sma_list = []
    closes = list(Series)
    for i in range(period,len(Series)): 
        sma_list.append(stat.mean(closes[i-period:i]))
    return [0]*period + sma_list

def makeStDevLst(df):
    period = 10
    closes = df['Close'].apply(float).tolist()
    stdev_list = []
    for i in range(period,len(df)):
        stdev_list.append(stat.pstdev(closes[i-period:i]*2)*2)
    return [0]*period + stdev_list

def laterPrice(period,df):
    temp_list = df['Close'].shift(-period).tolist()[0:-period] 
    return temp_list + [-1]*period

def makeStock(df,period):
    Stock_list_raw = []
    Stock_list = []
    closes = df['Close'].apply(float).tolist()
    highs = df['High'].apply(float).tolist()
    lows = df['Low'].apply(float).tolist()
    for i in range(period,len(df)):
        high = max(highs[i-period:i+1])
        low = min(lows[i-period:i+1])
        if (high-low) != 0:
            Stock_list_raw.append(((closes[i]-low)/(high-low) )*100)
        else:
            Stock_list_raw.append(Stock_list_raw[-1])
    for i in range(3,len(Stock_list_raw)):
        Stock_list.append(stat.mean(Stock_list_raw[i-3:i]))
    Stock = makeSMA(Stock_list,3)
    return [0]*(3+period) + Stock_list 

def makeEMA(s,n):
    s = s.tolist()
    ema = []
    j = 1
    sma = sum(s[:n]) / n
    multiplier = 2 / float(1 + n)
    ema.append(sma)
    ema.append(( (s[n] - sma) * multiplier) + sma)
    for i in s[n+1:]:
        tmp = ( (i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)
    return [0]*(n-1) + ema

def RSIDipped(Series):
    temp = []
    rsi_list = Series
    for i in range(7,len(rsi_list)):
        if len(rsi_list[i-7:i][rsi_list[i-7:i]<30]) != 0:
            temp.append(True)
        else:
            temp.append(False)
    return [False]*7 + temp

def RSIPeaked(Series):
    temp = []
    rsi_list = Series
    for i in range(7,len(rsi_list)):
        if len(rsi_list[i-7:i][rsi_list[i-7:i]>70]) != 0:
            temp.append(True)
        else:
            temp.append(False)
    return [False]*7 + temp

def StockDipped(Series):
    temp = []
    rsi_list = Series
    for i in range(7,len(rsi_list)):
        if len(rsi_list[i-7:i][rsi_list[i-10:i]<20]) != 0:
            temp.append(True)
        else:
            temp.append(False)
    return [False]*7 + temp

def StockPeaked(Series):
    temp = []
    rsi_list = Series
    for i in range(7,len(rsi_list)):
        if len(rsi_list[i-7:i][rsi_list[i-10:i]>80]) != 0:
            temp.append(True)
        else:
            temp.append(False)
    return [False] * 7 + temp

def newData():
    for i in data_dic.keys():
        if  (getdata(i,2,printtime = False)['Close'][-20:-3].reset_index(drop = True) == data_dic[i]['Close'][-20:-3].reset_index(drop = True)).all():
            return False
    return True

def updateData():
    starttime = time.time()
    temp_data_dic = {}
    for curr in crypto_list:

        tempdf = getdata(symbol = curr,daysback = 40,printtime = False)

        tempdf['Gain'] = tempdf['Close']>tempdf['Open']
        tempdf['Change'] = tempdf['Close']-tempdf['Open']
        tempdf['Percent Change'] = ((tempdf['Close']-tempdf['Open'])/tempdf['Open'])*100
        tempdf['12 EMA'] = pd.DataFrame(tempdf['Close']).ewm(span=12, min_periods=12).mean().values
        tempdf['26 EMA'] = pd.DataFrame(tempdf['Close']).ewm(span=26, min_periods=26).mean().values
        tempdf['MACD Line'] = [0]*25+((tempdf['12 EMA']-tempdf['26 EMA']).tolist()[25:])
        tempdf['Signal Line'] = pd.DataFrame(tempdf['MACD Line']).ewm(span=15, min_periods=15).mean().values
        tempdf['RSI'] = makeRSI(tempdf)
        tempdf['Fast Stock'] = makeStock(tempdf,3)
        tempdf['Slow Stock'] = makeStock(tempdf,14)
        tempdf['RSI Dipped'] = RSIDipped(tempdf['RSI'])
        tempdf['RSI Peaked'] = RSIPeaked(tempdf['RSI'])
        tempdf['Stock Dipped'] = RSIDipped(tempdf['Fast Stock'])
        tempdf['Stock Peaked'] = RSIPeaked(tempdf['Fast Stock'])
        temp_data_dic[curr] = tempdf.iloc[50:,:].reset_index(drop = True)
        
    print('Data Update Complete | ' + str(round(time.time()-starttime, 4)) + ' | ' + 
          str(round((time.time()-starttime)/len(crypto_list),4)))
    print()
    return temp_data_dic
    
def openPosition(c):
    if (getCoinbaseBalance(c)*getPrice(c)) > 6:
        return True
    else:
        return False

def sufficientFunds():
    return True

def openPositions():
    positions = 0
    for i in crypto_list:
        if openPosition(i):
            positions += 1
    return positions

def liquidate():
    for i in crypto_list:
        try:
            sell(i)
        except:
            print('Failed to Sell ' + i)
        time.sleep(.3)

def buy(curr):
    position_size = round(getCoinbaseBalance('USD') / (len(crypto_list)-openPositions()),2)
    try:
        client.place_market_order(product_id = curr + '-USD', side='buy', funds = str(position_size))
    except:
        print('Error Buying ' + curr)

def sell(curr):
    try:
        client.place_market_order(product_id = curr + '-USD', side='sell', funds = str(round(getCoinbaseBalance(curr)*getPrice(curr),2)))
    except:
        print('Error Selling ' + curr)
 
def getCoinbaseBalance(item):
    accounts = client.get_accounts()
    while True:
        if type(accounts) == type({}):
            accounts = client.get_accounts()
        else:
            break
        time.sleep(.3)
    
    for i in accounts:
        if i['currency'] == item.upper():
            return float(i['balance'])
        
def getPrice(tick):
    holder = client.get_product_ticker(tick+'-USD')
    while True:
        if 'message' in holder.keys():
            holder = client.get_product_ticker(tick+'-USD')
        else:
            break
        time.sleep(.3)
    return float(holder['price'])

def makeCurrentPrices(lst):
    lst = list(lst)
    price_list = []
    for c in lst:
        price_list.append(getPrice(c))
        time.sleep(.2)
    return price_list

def sendText(string):
    try:
        Twilioclient.messages.create(from_='+15153052518',
                                     to='4159262876',
                                     body=str(string))
    except:
        print('Error Sending Text')

# In[26]:

original_account_balance = getCoinbaseBalance('USD')

data_dic = updateData()

print('Setup Complete')
print()
print('--------------------------')
print()

# ALGORITHM (runs constantly)
while True:
    try:
        while True:
            try:
                starttime = time.time()
                for curr in crypto_list:
                    data = data_dic[curr]
                    data = data.iloc[-4:-3,:]
                    data = dict(data)
                    for i in data.keys():
                        data[i] = data[i].values[0]
                    if (openPosition(curr))&(data['Stock Dipped'] & (data['MACD Line'] > data['Signal Line']) & (data['RSI']<60)):
                        sell(curr)
                        print(curr.ljust(5) + ' Sell')
                        sendText('Sold ' + curr + ' at ' + str(getPrice(curr))) 
                    elif (openPosition(curr) == False)&(((data['MACD Line']<data['Signal Line']) & data['Stock Peaked'])|((data['MACD Line']<data['Signal Line'])&data['RSI Peaked'])):
                        buy(curr)
                        print(curr.ljust(5) + 'Buy')
                        sendText('Bought ' + curr + ' at ' + str(getPrice(curr)))
                    else:
                        print(curr.ljust(5) + ' N/A')
                print()
                print('Data Peek Complete | ' + 
                      str(round(time.time()-starttime, 4)) + ' | '  + 
                      str(round((time.time()-starttime)/len(crypto_list),4)))
                print()
                print('--------------------------')
                print()
                while newData() == False:
                    time.sleep(10)
                print('New Data Available ' + str(datetime.datetime.now())[:-10])
                print()
                data_dic = updateData()
            
            except:
                print('--- Error ---')
                time.sleep(10)
                time.sleep(15)
                      
    except simplejson.errors.JSONDecodeError:
        for i in range(5):
            print('-----------------------------')
        print('---JSON DECODING ERROR---')
        for i in range(5):
            print('-----------------------------')
        print('Sleeping for 15 Minutes')
        time.sleep(900)
    
    except:
        for i in range(5):
            print('-----------------------------')
        print('---UNKNOWN ERROR---')
        for i in range(5):
            print('-----------------------------')
        print('Sleeping for 15 Minutes')
        time.sleep(900)

# In[ ]:

# In[ ]: