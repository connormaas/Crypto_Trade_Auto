# Backtester (tests on historical data)
# NOTE: - Certain features have been excluded to prevent exposure of personal 
#         information

# In[1]:

import pandas as pd 
import pandas as pd
import yfinance as yf
import pyautogui as pag
import time 
import alpaca_trade_api as tradeapi
import statistics as stat
from matplotlib import pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from yahoo_fin import stock_info as si
import datetime
import scipy.interpolate as interpolate
from scipy.signal import argrelmax
from scipy.signal import argrelmin
from scipy.signal import find_peaks
import cbpro
import requests
import numpy as np
register_matplotlib_converters()

# In[2]:

crypto_list = ['BTC','XLM','ETH','XRP','EOS','LTC','BCH','ETC','LINK','REP','ZRX','KNC']

not_listed_cryptos = ['XTZ','ALGO','OXT','ATOM']
data_dic = {}

# In[3]:

def ljust(sarr):
    tempsarr = []
    for i in sarr:
        tempsarr.append(i.ljust(9))
    
    return tempsarr

def success_rate(lst):
    lst = np.array(list(lst))
    return len(lst[lst>0])/len(lst)
    

def getdata(symbol,interval = '1h',daysback = 5,unix = True):
    symbol = symbol.upper()
    dta = yf.download(tickers = symbol + '-USD', period = str(daysback)+'d' , interval = interval,progress = False).reset_index()
    if unix:
        dta['Time'] = pd.DatetimeIndex (dta['index']).astype(np.int64)/1000000
    else:
        dta['Time'] = pd.to_datetime(dta['index'])
    dta.drop(['index'],axis = 1,inplace = True)
    return dta  

def makeTimes(unix):
    date = datetime.datetime.utcfromtimestamp(unix)
    return date

def makeRSI(df):
    period = 7
    rsi_list = []
    for i in range(period,len(df)):
        tempdata = df[i-period:i]  
        if len(tempdata[tempdata['Gain']]['Change'])==0: 
            gains = 0
        else:          
            gains = stat.mean(tempdata[tempdata['Gain']]['Change'])    
        if len(tempdata[tempdata['Gain']==False]['Change'])==0:
            loses = 0
        else:    
            loses = stat.mean(tempdata[tempdata['Gain']==False]['Change'].apply(abs))
        if loses == 0:
            rsi_list.append(100)
        else:    
            rsi_list.append(100 - 100/(1+(gains/loses)))
            
    return [0]*period + rsi_list

def makeOBV(df):
    period = 10
    obv_list = []
    for i in range(period,len(df)):
        tempdata = df[i-period:i]
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

def makeStoch(df,period):
    stoch_list_raw = []
    stoch_list = []
    closes = df['Close'].apply(float).tolist()
    highs = df['High'].apply(float).tolist()
    lows = df['Low'].apply(float).tolist()
    for i in range(period,len(df)):
        high = max(highs[i-period:i+1])
        low = min(lows[i-period:i+1])
        if (high-low) != 0:
            stoch_list_raw.append(((closes[i]-low)/(high-low) )*100)
        else:
            stoch_list_raw.append(stoch_list_raw[-1])
    for i in range(3,len(stoch_list_raw)):
        stoch_list.append(stat.mean(stoch_list_raw[i-3:i]))
    stoch = makeSMA(stoch_list,3)
    return [0]*(3+period) + stoch_list 

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

def StochDipped(Series):
    temp = []
    rsi_list = Series
    for i in range(7,len(rsi_list)):
        if len(rsi_list[i-7:i][rsi_list[i-10:i]<20]) != 0:
            temp.append(True)
        else:
            temp.append(False)
    return [False]*7 + temp

def StochPeaked(Series):
    temp = []
    rsi_list = Series
    for i in range(7,len(rsi_list)):
        if len(rsi_list[i-7:i][rsi_list[i-10:i]>80]) != 0:
            temp.append(True)
        else:
            temp.append(False)
    return [False] * 7 + temp

# def getdata(symbol,daysback):
#     symbol = symbol.upper()
#     holder = client.get_product_historic_rates(symbol + '-USD',granularity=3600)
#     while type(holder) == type({}):
#         holder = client.get_product_historic_rates(symbol + '-USD',granularity=3600)
#     dta = pd.DataFrame(holder) 
#     dta.columns = ['Time','Low','High','Open','Close','Volume']
#     dta.sort_values(by = 'Time', ascending = True, inplace = True)
#     return dta.iloc[-daysback:,:]


# In[4]:

print(crypto_list)
print()
for curr in crypto_list:

    tempdf = getdata(symbol = curr,daysback = 300)
    
    tempdf = tempdf.sort_values(by = 'Time',ascending = True).reset_index()
    tempdf['Gain'] = tempdf['Close']>tempdf['Open']
    tempdf['Change'] = tempdf['Close']-tempdf['Open']
    tempdf['Percent Change'] = ((tempdf['Close']-tempdf['Open'])/tempdf['Open'])*100
    tempdf['12 EMA'] = pd.DataFrame(tempdf['Close']).ewm(span=12, min_periods=12).mean().values
    tempdf['26 EMA'] = pd.DataFrame(tempdf['Close']).ewm(span=26, min_periods=26).mean().values
    tempdf['MACD Line'] = [0]*25+((tempdf['12 EMA']-tempdf['26 EMA']).tolist()[25:])
    tempdf['Signal Line'] = pd.DataFrame(tempdf['MACD Line']).ewm(span=15, min_periods=15).mean().values
    tempdf['RSI'] = makeRSI(tempdf)
    tempdf['Fast Stoch'] = makeStoch(tempdf,3)
    tempdf['Slow Stoch'] = makeStoch(tempdf,14)
    tempdf['RSI Dipped'] = RSIDipped(tempdf['RSI'])
    tempdf['RSI Peaked'] = RSIPeaked(tempdf['RSI'])
    tempdf['Stoch Dipped'] = RSIDipped(tempdf['Fast Stoch'])
    tempdf['Stoch Peaked'] = RSIPeaked(tempdf['Fast Stoch'])
    data_dic[curr] = tempdf.iloc[50:,:].reset_index(drop = True)
    print(curr + ' Done')
    
print('--- Done ---')


# In[ ]:


def makeBuySellIndex(df):
    #sell_ind   = pd.Series(range(len(df))).to_numpy()[df['Stoch Dipped'] & (df['MACD Line']>df['Signal Line'])&(df['RSI']<60)]
    #buy_ind = pd.Series(range(len(df))).to_numpy()[((df['MACD Line']<df['Signal Line'])&df['Stoch Peaked'])|((df['MACD Line']<df['Signal Line'])&df['RSI Peaked'])]
    
    sell_ind = np.array(df.index)[df['Stoch Dipped'] & (df['MACD Line']>df['Signal Line'])&(df['RSI']<60)]
    buy_ind = np.array(df.index)[((df['MACD Line']<df['Signal Line'])&df['Stoch Peaked'])|((df['MACD Line']<df['Signal Line'])&df['RSI Peaked'])]

    #sell_ind   = pd.Series(range(len(df))).to_numpy()[df['Stoch Dipped'] & (df['MACD Line']>df['Signal Line'])&(df['RSI']<60)]
    #buy_ind = pd.Series(range(len(df))).to_numpy()[((df['MACD Line']<df['Signal Line'])&df['Stoch Peaked'])|((df['MACD Line']<df['Signal Line'])&df['RSI Peaked'])]


#     sell_ind   = pd.Series(range(len(df))).to_numpy()[df['Stoch Dipped'] & (df['MACD Line']>df['Signal Line'])&(df['RSI']<60)]
#     buy_ind = pd.Series(range(len(df))).to_numpy()[((df['MACD Line']<df['Signal Line'])&df['Stoch Peaked']&(df['Fast Stoch']<20))|((df['MACD Line']<df['Signal Line'])&df['RSI Peaked']&(df['RSI']<30))]


    buy_ind = np.array(list(pd.Series(buy_ind)+2))
    sell_ind = np.array(list(pd.Series(sell_ind)+2))
    
    
    return {'buy':buy_ind,'sell':sell_ind}


# In[ ]:


stop_loss = 1.0
position_size = 1000
fee_size = .0025

show_graphs = True
show_stats = True


diversified_position_size = 1000/len(data_dic)
coin_counts = {}
diversified_position_values = {}
for i in data_dic.keys():
    coin_counts[i] = diversified_position_size/(data_dic[i]['Close'].tolist()[0])
for i in data_dic.keys():
    prices = data_dic[i]['Close']
    diversified_position_values[i] = prices*coin_counts[i]
total_portfolio_values = diversified_position_values[(list(diversified_position_values.keys())[0])]
for i in list(diversified_position_values.keys())[1:]:
    total_portfolio_values = total_portfolio_values+diversified_position_values[i]
total_portfolio_values = (total_portfolio_values)

standard_devs = []
success_rates = []
Total_Net_Profit = []
Total_Trades = []
Average_Distance_From_Low = []
Stopped_out = []
average_gains = []
average_losses = []
average_loss_prevented = []
buy_index_by_currency = {}
sell_index_by_currency = {}

sp500_data = yf.download(tickers = 'SPY',interval = '1h', period = '730d')
shares = 1000/sp500_data['Close'][0]
sp500_buy_and_hold = sp500_data['Close']*shares
currency_lists = {}
currency_lists['Volatile'] = ['XLM','XRP','EOS','LINK']
currency_lists['Stable'] = ['BTC','ETH','LTC']
currency_lists['All'] = ['XLM','BTC','ETH','XRP','EOS','LINK', 'LTC','REP','KNC']

for currency in crypto_list:
    data = data_dic[currency]
    bs_index = makeBuySellIndex(data)
    buy_index   = bs_index['buy']
    sell_index = bs_index['sell']

    sell_index = sell_index[sell_index < len(data)-2]
    buy_index = buy_index[buy_index < sell_index[-1]]

    metric = 'Close'

    profit = []
    real_profit = []
    place = -1
    real_buy_ind = []
    real_sell_ind = []
    time_held = []
    prices = list(data['Close'])
    real_profit_no_stop = []
    for i in buy_index:
        if sell_index[sell_index>i][0] > place:
            buy = i
            sell = sell_index[sell_index>i][0]
            buy_price = data['Close'][i]
            sell_price = data['Close'][sell_index[sell_index>i][0]]

            real_buy_ind.append(i)
            real_sell_ind.append(sell_index[sell_index>i][0])
            time_held.append(sell_index[sell_index>i][0] - i)
            place = sell_index[sell_index>i][0]
            if ((prices[buy]-min(prices[buy:sell]))/prices[buy])>stop_loss:
                profit.append(buy_price*(-stop_loss))
                real_profit.append((-stop_loss)* position_size)
            else:
                profit.append(sell_price-buy_price)
                real_profit.append(((sell_price-buy_price)/buy_price )* position_size)
            real_profit_no_stop.append(((sell_price-buy_price)/buy_price )* position_size)

    profit = np.array(profit)
    real_profit = np.array(real_profit)
    buy_index_by_currency[currency] = real_buy_ind
    sell_index_by_currency[currency] = real_sell_ind

    success_rates.append(round(100*(len(real_profit[(pd.Series(real_profit)-(position_size*fee_size))>0])/len(real_profit)),2))
    standard_devs.append(round(stat.pstdev(data['Close'].tolist())/stat.mean(data['Close'].tolist()),4))
    Total_Net_Profit.append(pd.Series(real_profit).sum() - (position_size*fee_size*len(profit)))
    Total_Trades.append(len(profit))
    mins = []
    for i in range(len(real_buy_ind)):
        mins.append((prices[real_buy_ind[i]]-min(prices[real_buy_ind[i]:real_sell_ind[i]]))/prices[real_buy_ind[i]])
    Average_Distance_From_Low.append(stat.mean(mins))
    mins = np.array(mins)
    Stopped_out.append(len(mins[mins > stop_loss])/len(mins))
    try:
        average_gains.append(stat.mean(pd.Series(real_profit[real_profit>0])))
    except:
        average_gains.append(0)
    
    try:
        average_losses.append(stat.mean(pd.Series(real_profit[real_profit<0])))
    except:
        average_losses.append(0)
    
    try:
        average_loss_prevented.append(stat.mean((real_profit-real_profit_no_stop)/position_size))
    except:
        average_loss_prevented.append(0)
    
    if show_stats:
        col1 = []
        col2 = []
        col1.append('Profit with $'+str(position_size)+' Positions:')
        col2.append(str(round(pd.Series(real_profit).sum(),4)))
        col1.append('Transaction Fees: ')
        col2.append(str(round(position_size*fee_size*len(profit),4)))
        col1.append('Trades Made:')
        col2.append(str(len(profit)))
        col1.append('Success Rate w/o Fees: ')
        col2.append(str(round(100*(len(profit[profit>0])/len(profit)),4)) + '%')
        col1.append('Success Rate w/ Fees: ')
        col2.append(str(round(100*(len(real_profit[(pd.Series(real_profit)-(position_size*fee_size))>0])/len(real_profit)),4)) + '%')
        
        col1.append('Average Gain: ')
        try:
            col2.append(str(round(stat.mean(pd.Series(real_profit[real_profit>0])),4)))
        except:
            col2.append(0)
        
        col1.append('Average Loss: ')
        try:
            col2.append(str(round(stat.mean(pd.Series(real_profit[real_profit<0])),4)))
        except:
            col2.append(0)
        
        col1.append('Average Time Held: ' )
        try:
            col2.append(str(stat.mean(time_held)))
        except:
            col2.append(0)
        
        col1.append('Standard Deviation: ' )
        col2.append(str(round(stat.pstdev(data['Close'].tolist())/stat.mean(data['Close'].tolist()),4)))
        col1.append('Average Distance From Bottom: ' )
        col2.append(str(round(stat.mean(mins),4)))
        col1.append('Percent Trades Stopped Out: ' )
        col2.append(str(round(100*(len(mins[mins > stop_loss])/len(mins)),4)))
        col1.append('Average Loss Prevented: ' )
        col2.append(stat.mean(real_profit_no_stop - real_profit))
        col1.append('Net Profit: ' )
        col2.append(str(round(pd.Series(real_profit).sum() - (position_size*fee_size*len(profit)),4)))
        print('--- ' + currency + ' ---')
        stats = pd.DataFrame(col2,col1)
        stats.columns = ['']
        print(stats)
    if show_graphs:
        clrs = []
        for i in list(pd.Series(real_profit) - (position_size*fee_size)):
            if i >0:
                clrs.append('green')
            else:
                clrs.append('red')

        print()
        account_values = []
        for i in range(len(real_profit)):
            account_values.append(sum(real_profit[0:i])-(position_size*fee_size*i))

        plt.figure(figsize = (20,7))
        plt.style.use('dark_background')
        plt.bar(list(range(0,len(real_profit))),pd.Series(real_profit) - (position_size*fee_size), color = clrs)
        plt.grid(True,alpha = .3)
        plt.title(currency + ': Historical Gains w/ Fees and Stop Loss')
        plt.show()
        plt.close()
        
        clrs = []
        for i in list(pd.Series(real_profit_no_stop) - (position_size*fee_size)):
            if i >0:
                clrs.append('green')
            else:
                clrs.append('red')

        plt.figure(figsize = (20,7))
        plt.style.use('dark_background')
        plt.bar(list(range(0,len(real_profit_no_stop))),pd.Series(real_profit_no_stop) - (position_size*fee_size), color = clrs)
        plt.grid(True,alpha = .3)
        plt.title(currency + ': Historical Gains w/ Fees No Stop Loss')
        plt.show()
        plt.close()

        coins = 1000/data['Close'].tolist()[0]

        plt.figure(figsize = (20,7))
        plt.style.use('dark_background')
        plt.plot(data['Time'][real_buy_ind],(pd.Series(account_values)+1000)-(position_size*fee_size), color = 'red',label = 'Trading Strategy')
        plt.plot(data['Time'][real_buy_ind],sp500_buy_and_hold[:len(profit)], color = 'b',label = 'S&P 500')
        plt.plot(data['Time'][real_buy_ind],(data['Close']*coins)[real_buy_ind], color = 'y',label = currency + ' Buy and Hold')
        plt.plot(data['Time'][real_buy_ind][:-22],total_portfolio_values[real_buy_ind][:-22], color = 'g',label = 'Diversified Crypto Portfolio')
        plt.xticks(data['Time'][real_buy_ind][0::100],(data['Time'][real_buy_ind]/1000).apply(round).apply(makeTimes)[0::100].apply(str).str.split(' ').str[0])
        plt.legend()
        plt.grid(True,alpha = .3)
        plt.title(currency + ': Account Values Over Trade History')
        plt.show()
        plt.close()

        plt.style.use('dark_background')
        show_recent_trades = len(real_profit)
        plt.figure(figsize = (20,7))
        plt.plot(list(data.index)[(data.index[real_buy_ind][-show_recent_trades]):],list(data['Close'])[(data.index[real_buy_ind][-show_recent_trades]):],color = 'w',linewidth = .6,alpha = .6)
        plt.scatter(list(data.index[real_buy_ind])[-show_recent_trades:],list(data['Close'][real_buy_ind])[-show_recent_trades:],label = 'Buy',color = 'Green',s = 25)
        plt.scatter(list(data.index[real_sell_ind])[-show_recent_trades:],list(data['Close'][real_sell_ind])[-show_recent_trades:],label = 'Sell',color = 'Red', s = 25)
        plt.title(currency + ': Last ' + str(show_recent_trades) + ' Trades')
        plt.xlim(list(data.index)[(data.index[real_buy_ind][-show_recent_trades]):][0]-1,list(data.index)[(data.index[real_buy_ind][-show_recent_trades]):][-1]+5)
        plt.grid(True,alpha = .3)
        plt.legend()
        plt.show()
        plt.close()
        print('Profit Last '+str(show_recent_trades)+' Trades $' + str(round(sum(list(pd.Series(np.array(real_profit)[-show_recent_trades:])-(position_size*fee_size))),2)))
        print('Market Profit ' + str(((data['Close'][real_buy_ind].tolist()[-1] - data['Close'][real_buy_ind].tolist()[-show_recent_trades])/data['Close'][real_buy_ind].tolist()[-show_recent_trades])*position_size))
        print('Last 20 Trades Executed in '  + str((data['Time'].tolist()[-1] - data['Time'][(data.index[real_buy_ind][-show_recent_trades])])/86400000) + ' Days')
        print('Beat Market by ' + str((round(sum(list(pd.Series(np.array(real_profit)[-show_recent_trades:])-((position_size*fee_size)))),2) -  ((data['Close'][real_buy_ind].tolist()[-1] - data['Close'][real_buy_ind].tolist()[-show_recent_trades])/data['Close'][real_buy_ind].tolist()[-show_recent_trades])*position_size)))
        print()
        
        # plt.style.use('dark_background')
        # show_recent_trades = 250
        # plt.figure(figsize = (20,7))
        # plt.plot(list(data.index)[(data.index[buy_index][-show_recent_trades]):],list(data['Close'])[(data.index[buy_index][-show_recent_trades]):],color = 'w',linewidth = .6,alpha = .6)
        # plt.scatter(list(data.index[buy_index])[-show_recent_trades:],list(data['Close'][buy_index])[-show_recent_trades:],label = 'Buy',color = 'Green',s = 25)
        # plt.scatter(list(data.index[sell_index])[-show_recent_trades:],list(data['Close'][sell_index])[-show_recent_trades:],label = 'Sell',color = 'Red', s = 25)
        # plt.title(currency + ': Last ' + str(show_recent_trades) + ' Signals')
        # plt.xlim(list(data.index)[(data.index[buy_index][-show_recent_trades]):][0]-1,list(data.index)[(data.index[buy_index][-show_recent_trades]):][-1]+5)
        # plt.grid(True,alpha = .3)
        # plt.legend()
        # plt.show()
        # plt.close()
          
    print()
    print()
        
print()
print()
print()
print('Average Profit '.ljust(40,'-'),stat.mean(Total_Net_Profit))
print('Average Success Rate '.ljust(40,'-') , str(stat.mean(success_rates)) + '%')
print('Average Hours Per Trade '.ljust(40,'-'),(len(data_dic['BTC']))/stat.mean(Total_Trades))
print('Average Profit Per Trade '.ljust(40,'-') ,stat.mean(Total_Net_Profit)/stat.mean(Total_Trades))
print('Average Distance From Bottom '.ljust(40,'-') ,stat.mean(Average_Distance_From_Low))
print('Average Percent Of Trades Stopped Out '.ljust(40,'-') ,round(stat.mean(Stopped_out),4))
print('Average Gain '.ljust(40,'-') ,stat.mean(average_gains))
print('Average Loss '.ljust(40,'-') ,stat.mean(average_losses))
print('Average Loss Prevented'.ljust(40,'-') ,stat.mean(average_loss_prevented))
print()
print()
plt.bar(list(range(len(Total_Net_Profit))),Total_Net_Profit)

# In[ ]:

success_rates_overall = []
profits_overall = []
average_losses_overall = []
average_gains_overall = []
average_trades_overall = []
no_drop = 0
for test_drop_counter in range(113):
    try:
        test_drop = test_drop_counter/1000
        stop_loss = .1
        position_size = 1000
        fee_size = .007
        drop_real_profit = []
        success_rates_with_drop = []
        trades_with_drop = []
        average_gains = []
        average_losses = []
        
        for currency in ['XLM','BTC','ETH','XRP','EOS','LINK', 'LTC']:
            data = data_dic[currency]
            bs_index = makeBuySellIndex(data)
            buy_index   = bs_index['buy']
            sell_index = bs_index['sell']
            sell_index = sell_index[sell_index < len(data)-2]
            buy_index = buy_index[buy_index < sell_index[-1]]
            metric = 'Close'

            profit = []
            real_profit = []
            place = -1
            real_buy_ind = []
            real_sell_ind = []
            time_held = []

            drop_real_profit = []
            total_profits_with_drop = []

            prices = list(data['Close'])

            for i in buy_index:
                if sell_index[sell_index>i][0] > place:
                    buy = i
                    sell = sell_index[sell_index>i][0]
                    buy_price = data['Close'][buy]
                    sell_price = data['Close'][sell]

                    if ((prices[buy]-min(prices[buy:sell]))/prices[buy])>test_drop:
                        drop_real_profit.append(((sell_price - (buy_price*(1-test_drop)))/(buy_price*(1-test_drop)))*position_size )

            success_rates_with_drop.append(round(100*success_rate(drop_real_profit),4))

            mins = []
            drop_real_profit = np.array(drop_real_profit)
            average_gains.append(stat.mean(pd.Series(drop_real_profit[drop_real_profit>0])))
            average_losses.append(stat.mean(pd.Series(drop_real_profit[drop_real_profit<0])*-1))
            total_profits_with_drop.append(sum(drop_real_profit)/len(['XLM','BTC','ETH','XRP','EOS','LINK', 'LTC']))
            trades_with_drop.append(len(drop_real_profit)/len(['XLM','BTC','ETH','XRP','EOS','LINK', 'LTC']))

        success_rates_overall.append(stat.mean(success_rates_with_drop))
        profits_overall.append(stat.mean(total_profits_with_drop))
        average_losses_overall.append(stat.mean(average_losses))
        average_gains_overall.append(stat.mean(average_gains))
        average_trades_overall.append(stat.mean(trades_with_drop))
        if test_drop*1000 % 10 == 0:
            print(str(test_drop) + '  Done')
    except:
        no_drop+=1

# In[ ]:

df_drops = pd.DataFrame({'Drop Tested':pd.Series(list(range(113-no_drop)))/10,'Success Rate':(success_rates_overall),'Profit':(profits_overall),'Average Loss':(average_losses_overall),'Average Gain':(average_gains_overall),'Average Trades':average_trades_overall})
df_drops

# In[ ]:

plt.figure(figsize = (15,7))
pd.Series(success_rates,standard_devs).sort_index().plot()
plt.xlabel('Volatility')
plt.ylabel('% Trades Won')
plt.title('Volatility vs. Success Rate')
plt.grid(True,alpha = .3)
plt.show()
plt.close()

# In[ ]:

plt.figure(figsize = (15,7))
plt.plot(df_drops['Drop Tested'], df_drops['Success Rate'])
plt.xlabel('% Drop From Original Buy Signal')
plt.ylabel('Success Rate')
plt.title('Drop vs. Success Rate')
plt.grid(True,alpha = .3)
plt.show()
plt.close()

# In[ ]:

plt.figure(figsize = (15,7))
plt.plot(df_drops['Drop Tested'], df_drops['Profit'])
plt.xlabel('% Drop From Original Buy Signal')
plt.ylabel('Profit')
plt.title('Drop vs. Profit')
plt.grid(True,alpha = .3)
plt.show()
plt.close()


# In[ ]:

plt.figure(figsize = (15,7))
plt.plot(df_drops['Drop Tested'], df_drops['Average Gain'])
plt.xlabel('% Drop From Original Buy Signal')
plt.ylabel('Average Gain')
plt.title('Drop vs. Average Gain')
plt.grid(True,alpha = .3)
plt.show()
plt.close()

# In[ ]:

plt.figure(figsize = (15,7))
plt.plot(df_drops['Drop Tested'], df_drops['Average Loss'])
plt.xlabel('% Drop From Original Buy Signal')
plt.ylabel('Average Loss')
plt.title('Drop vs. Average Loss')
plt.grid(True,alpha = .3)
plt.show()
plt.close()

# In[ ]:

plt.figure(figsize = (15,7))
plt.plot(df_drops['Drop Tested'], df_drops['Average Trades'])
plt.xlabel('% Drop From Original Buy Signal')
plt.ylabel('Average Trades')
plt.title('Drop vs. Average Trades')
plt.grid(True,alpha = .3)
plt.show()
plt.close()

# In[ ]:

# In[ ]:

if curr not in not_listed_cryptos:
        tempdf = getdata(interval = '1h',symbol = curr,daysback = 100,unix = True)
else:
    tempdf = pd.DataFrame(client.get_product_historic_rates(curr + '-USD',granularity=3600)) 
    tempdf.columns = ['Time','Low','High','Open','Close','Volume']

tempdf = tempdf.sort_values(by = 'Time',ascending = True).reset_index()
tempdf = tempdf.iloc[:-86,:]

tempdf['Gain'] = tempdf['Close']>tempdf['Open']
tempdf['Change'] = tempdf['Close']-tempdf['Open']
tempdf['Percent Change'] = ((tempdf['Close']-tempdf['Open'])/tempdf['Open'])*100
tempdf['12 EMA'] = pd.DataFrame(tempdf['Close']).ewm(span=12, min_periods=12).mean().values
tempdf['26 EMA'] = pd.DataFrame(tempdf['Close']).ewm(span=26, min_periods=26).mean().values
tempdf['MACD Line'] = [0]*25+((tempdf['12 EMA']-tempdf['26 EMA']).tolist()[25:])
tempdf['Signal Line'] = pd.DataFrame(tempdf['MACD Line']).ewm(span=15, min_periods=15).mean().values
tempdf['RSI'] = makeRSI(tempdf)
tempdf['Fast Stoch'] = makeStoch(tempdf,3)
tempdf['Slow Stoch'] = makeStoch(tempdf,14)
tempdf['RSI Dipped'] = RSIDipped(tempdf['RSI'])
tempdf['RSI Peaked'] = RSIPeaked(tempdf['RSI'])
tempdf['Stoch Dipped'] = RSIDipped(tempdf['Fast Stoch'])
tempdf['Stoch Peaked'] = RSIPeaked(tempdf['Fast Stoch'])
td1 = tempdf

# In[ ]:

if curr not in not_listed_cryptos:
        tempdf = getdata(interval = '1h',symbol = curr,daysback = 100,unix = True)
else:
    tempdf = pd.DataFrame(client.get_product_historic_rates(curr + '-USD',granularity=3600)) 
    tempdf.columns = ['Time','Low','High','Open','Close','Volume']
tempdf = tempdf.sort_values(by = 'Time',ascending = True).reset_index()
tempdf['Gain'] = tempdf['Close']>tempdf['Open']
tempdf['Change'] = tempdf['Close']-tempdf['Open']
tempdf['Percent Change'] = ((tempdf['Close']-tempdf['Open'])/tempdf['Open'])*100
tempdf['12 EMA'] = pd.DataFrame(tempdf['Close']).ewm(span=12, min_periods=12).mean().values
tempdf['26 EMA'] = pd.DataFrame(tempdf['Close']).ewm(span=26, min_periods=26).mean().values
tempdf['MACD Line'] = [0]*25+((tempdf['12 EMA']-tempdf['26 EMA']).tolist()[25:])
tempdf['Signal Line'] = pd.DataFrame(tempdf['MACD Line']).ewm(span=15, min_periods=15).mean().values
tempdf['RSI'] = makeRSI(tempdf)
tempdf['Fast Stoch'] = makeStoch(tempdf,3)
tempdf['Slow Stoch'] = makeStoch(tempdf,14)
tempdf['RSI Dipped'] = RSIDipped(tempdf['RSI'])
tempdf['RSI Peaked'] = RSIPeaked(tempdf['RSI'])
tempdf['Stoch Dipped'] = RSIDipped(tempdf['Fast Stoch'])
tempdf['Stoch Peaked'] = RSIPeaked(tempdf['Fast Stoch'])
td2 = tempdf


# In[ ]:

tl = td2.iloc[:103,:]

# In[ ]:

td2.iloc[100:110,:]

# In[ ]:

for i in range(len(after)):
    print(str(i) + '  ' +str(before[i] == after[i]))

# In[ ]:

tl = getdata(interval = '1h',symbol = curr,daysback = 100,unix = True)
tl['Gain'] = tl['Close']>tl['Open']
tl['Change'] = tl['Close']-tl['Open']

# In[ ]:

tl['RSI 1'] = makeRSI(tl)

# In[ ]:

tl['RSI 2'] = makeRSI(tl.iloc[:-50,:])+[-1]*50

# In[ ]:

tl['12 EMA'] = pd.DataFrame(tl['Close']).ewm(span=12, min_periods=12).mean().values

# In[ ]:

tl['12 EMA 2'] = list(pd.DataFrame(tl['Close'][:-50]).ewm(span=12, min_periods=12).mean().values )+ [-1]*50

# In[ ]:

makeBuySellIndex(data_dic['BTC'])['buy'][-5:]

# In[ ]:

data_dic['BTC'].index[-1]+3

# In[ ]:

getdata(symbol = "BTC",daysback = 5)

# In[ ]:

dd = yf.download(tickers = 'BTC-USD', period = str(5)+'d',progress = False,interval = "1h").reset_index()
dd.head()

# In[ ]:

dd["index"]

# In[ ]: