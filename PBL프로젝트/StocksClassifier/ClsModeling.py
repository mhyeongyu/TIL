import pandas as pd
import Classifier as cls

names_lst = ['삼성전자', 'SK하이닉스', 'LG화학', '카카오', 'NAVER', '현대차']
days_lst = [5, 20, 60, 120]

stock_code = pd.read_csv('./KOSPI_200.csv', dtype={'종목코드': str, '종목명': str})[['종목명', '종목코드']]

for name in names_lst:
    print(f'Start {name} Modeling!!')
    data, code = cls.load_stocks_data(f'{name}', stock_code)

    for day in days_lst:
        stocks = cls.Stocks(data)
        stocks.preprocessing()
        sign_data = stocks.stocksign(stocks.data, day)
        para = stocks.modeling(sign_data, code, day)
        parameter = stocks.modelings(sign_data, code, day)