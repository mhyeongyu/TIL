import numpy as np
import pandas as pd
import Regression as rr
import FinanceDataReader as fdr
from datetime import timedelta, date
from tqdm import tqdm

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def hola_sec_function(name, day, alpha):

    stock_code = pd.read_csv('./KOSPI_200.csv', dtype={'종목코드': str, '종목명': str})[['종목명', '종목코드']]
    data, code = rr.load_stocks_data(f'{name}', stock_code)
    stocks = rr.Stocks(data)
    result = stocks.predict(code, int(day), alpha)
    
    result = np.array(result[:-1])
    test = fdr.DataReader(f'{code}', date.today() - timedelta(days=29), date.today())['Close'].values
    mape = mean_absolute_percentage_error(test, result)

    return mape

names_lst = ['삼성전자', 'SK하이닉스', 'LG화학', '카카오', 'NAVER', '현대차']
# names_lst = ['삼성전자']
days_lst = [20]
alpha_lst = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

result_df = pd.DataFrame(columns=['name', 'alpha', 'mape'])

for name in tqdm(names_lst):
    for day in days_lst:
        for a in alpha_lst:
            mape = hola_sec_function(name, day, a)
            
            result_df = result_df.append({'name': name,
                                          'alpha': a,
                                          'mape': mape}, ignore_index=True)

print(result_df)
result_df.to_csv('result.csv', index=False)