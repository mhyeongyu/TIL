import pandas as pd
import Classifier as cls

def hola_sec_function(name='삼성전자', day=5):

    # 종목명, 종목코드 불러오기
    stock_code = pd.read_csv('./KOSPI_200.csv', dtype={'종목코드': str, '종목명': str})[['종목명', '종목코드']]

    data, code = cls.load_stocks_data(f'{name}', stock_code)
    stocks = cls.Stocks(data)
    stocks.preprocessing()
    sign_data = stocks.stocksign(stocks.data, int(day))
    result = stocks.predict(sign_data, code, int(day))

    return result

names_lst = ['삼성전자', 'SK하이닉스', 'LG화학', '카카오', 'NAVER', '현대차']
days_lst = [5, 20, 60, 120]
result_df = pd.DataFrame(columns=['name', 'day', 'predict'])

for name in names_lst:
    for day in days_lst:
        answer = hola_sec_function(name, day)
        result_df = result_df.append({'name': name,
                                      'day': day,
                                      'predict': answer}, ignore_index=True)

print(result_df)