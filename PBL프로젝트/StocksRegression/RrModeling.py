import pandas as pd
import Regression as rr

names_lst = ['삼성전자', 'SK하이닉스', 'LG화학', '카카오', 'NAVER', '현대차']
days_lst = [5, 20, 60, 120]

stock_code = pd.read_csv('./KOSPI_200.csv', dtype={'종목코드': str, '종목명': str})[['종목명', '종목코드']]

for name in names_lst:
    print(f'Start {name} Modeling!!')
    data, code = rr.load_stocks_data(f'{name}', stock_code)

    for day in days_lst:
        stocks = rr.Stocks(data)
        stocks.modeling(code, day)