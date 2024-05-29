import  os
import time
import redis
import numpy as np
import pandas as pd
import pandas_ta as ta
import json
import re
import atexit

from htrfnwe import (
    ewma,
    ema2,
    tema3,
    halftrend,
    run_nwe,
    ema,
    range_size,
    range_filter,
    vumanchu_swing,
)
class DataFrameHandler:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def convert_data_types(data:pd.DataFrame):
        dtypes_map = {
            'Low': 'float32',
            'Close': 'float32',
            'Open': 'float32',
            'High': 'float32',
            'oi': 'float32',
            'TR': 'float32',
            'pHigh': 'float32',
            'pLow': 'float32',
            'token': 'category',
            'symbol': 'category',
            'strprc': 'float32',
            'Type': 'category',
            'Date': 'datetime64[ns]',
            'Time': 'datetime64[ns]'
        }
        return data.astype(dtypes_map)

class TradingStrategy:
    import pandas as pd
    def __init__(self, host='localhost', port=6379):
        self.host = host
        self.port = port
    def _connect_redis(self):
        try:
            r = redis.StrictRedis(host=self.host, port=self.port, decode_responses=True)
            return r
        except Exception as e:
            raise Exception(f"Error connecting to Redis: {e}")        
    def fetch_data_from_redis(self,keys_prefix, n) -> pd.DataFrame:
        r = self._connect_redis()
        reversed_df=pd.DataFrame(None)
        data=r.execute_command(f"TS.MREVRANGE - + WITHLABELS COUNT {n} FILTER timeframe={keys_prefix.split(':')[-1]} c2={keys_prefix.split(':')[-2]}")
        # Extracting data into dictionary
        ohlc_data = {}
        for item in data:
            key = item[0].split(':')[-1]  # Extracting OHLC key
            values = item[2]  # Extracting timestamp and value pairs
            ohlc_data[key] = values

        # Converting timestamp to datetime and values to float
        for key, values in ohlc_data.items():
            timestamps = [pd.to_datetime(ts, unit='ms',utc=True).tz_convert('Asia/Kolkata').tz_localize(None) for ts, _ in values]
            #values = [float(val) for _, val in values]
            values=np.array([np.float32(val) for _, val in values], dtype=np.float32)
            ohlc_data[key] = {'Time': timestamps, key: values}

        # Creating DataFrame
        reversed_df = pd.DataFrame(ohlc_data['Open'])
        reversed_df.set_index('Time', inplace=True)  # Setting Timestamp as index

        # Adding other columns to DataFrame
        for key in ['High', 'Low', 'Close', 'oi', 'volume']:
            reversed_df[key] = ohlc_data[key][key]

        reversed_df.sort_index(ascending=True,inplace=True)
        reversed_df['pHigh'] = reversed_df['High'].rolling(3).apply(lambda x: np.nanmax(x[:-1]), raw=True).shift(-1).fillna(axis=0,method='ffill')
        reversed_df['pLow'] = reversed_df['Low'].rolling(3).apply(lambda x: np.nanmin(x[:-1]), raw=True).shift(-1).fillna(axis=0,method='ffill')
        reversed_df['TR_High'] = ta.true_range(high=reversed_df['High'], low=reversed_df['Open'], close=reversed_df['pHigh'],talib=True).fillna(axis=0,method='ffill')
        reversed_df['TR_Low'] = ta.true_range(high=reversed_df['pLow'], low=reversed_df['Low'], close=reversed_df['Close'],talib=True).fillna(axis=0,method='ffill')
        
        reversed_df['TR']=reversed_df['TR_High']*0.5+reversed_df['TR_Low']*0.5
        reversed_df.dropna(inplace=True)
      
        return pd.DataFrame(reversed_df.convert_dtypes())


keys_prefix="markets:56656:banknifty29may24p48900:1s"
#def fetch_data_from_redis(self,keys_prefix, n) -> pd.DataFrame:
r = TradingStrategy()._connect_redis()
reversed_df=pd.DataFrame(None)
reversed_df=TradingStrategy().fetch_data_from_redis(keys_prefix=keys_prefix,n=300)
key_parts = keys_prefix.split(":")
reversed_df['token'] = key_parts[1]
reversed_df['symbol'] = key_parts[2].upper()

try:
    reversed_df['strprc'] = np.float32(re.split('24c|24C|24P|24p', key_parts[2])[1])
    reversed_df['Type'] = (lambda x: 'CE' if re.search(r'24[Cc]', x) else ('PE' if re.search(r'24[Pp]', x) else None))(key_parts[2])
except (ValueError, IndexError):
    reversed_df['strprc'] = reversed_df['Close'].astype(np.float32)
    reversed_df['symbol'] = 'NiftyBank'
    reversed_df['Type'] = 'Index'
    
reversed_df['Date'] = pd.to_datetime(reversed_df.index)
reversed_df['Time'] = pd.to_datetime(reversed_df.index)
reversed_df=DataFrameHandler(reversed_df).convert_data_types(reversed_df)       
reversed_df['rf_High']= vumanchu_swing(reversed_df['High'].to_numpy(dtype=np.double),swing_period=3,swing_multiplier=0.75)[:, 2]
reversed_df['rf_Low']= vumanchu_swing(reversed_df['Low'].to_numpy(dtype=np.double),swing_period=2,swing_multiplier=0.75)[:, 2]
reversed_df['halftrend']=halftrend(reversed_df['High'].to_numpy(dtype=np.float32),reversed_df['Low'].to_numpy(dtype=np.float32),reversed_df['Close'].to_numpy(dtype=np.float32),reversed_df['TR'].to_numpy(dtype=np.float32),amplitude=1,channel_deviation=0.786)['halftrend']
reversed_df['nwe_NWEHigh']=pd.DataFrame(run_nwe(reversed_df[['High']].to_numpy(dtype=np.float32)), index=reversed_df.index)
reversed_df['nwe_NWELow']=pd.DataFrame(run_nwe(reversed_df[['Low']].to_numpy(dtype=np.float32)), index=reversed_df.index)

reversed_df['nweavg'] = (reversed_df['nwe_NWEHigh']*.50 + reversed_df['nwe_NWELow']*.50) # applying nweavg crossover straegy become suprime but have doubt it will repaint nweavg signal because of calculation.
reversed_df['rf'] = (reversed_df['rf_High']*.50 + reversed_df['rf_Low']*.50).astype(np.float32)
reversed_df['halftrend'] = (reversed_df['nweavg']*.20+ reversed_df['halftrend']*.80) # good sucess
print(reversed_df)
