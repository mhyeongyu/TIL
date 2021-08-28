import pandas as pd
import Classifier as cls
from tqdm import tqdm

stock_code = pd.read_csv('./KOSPI_100.csv', dtype={'종목코드': str, '종목명': str})[['종목명', '종목코드']]

names_lst = list(stock_code['종목명'].values)
days_lst = [5, 20, 60, 120]

for name in tqdm(names_lst):
    try:
        print(f'\nStart {name} Modeling!!')
        data, code = cls.load_stocks_data(f'{name}', stock_code)
        stocks = cls.Stocks(data)
        stocks.preprocessing()
        
        for day in days_lst:

            sign_data = stocks.stocksign(stocks.data, day)
            para = stocks.xgb_modeling(sign_data, code, day)
            parameter = stocks.rf_modeling(sign_data, code, day)
        
    except:
        print(f'{name} ERROR!!')

print('\nFinish!!!!')