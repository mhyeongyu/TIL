import pandas as pd
import Regression as rr
from tqdm import tqdm

stock_code = pd.read_csv('./KOSPI_100.csv', dtype={'종목코드': str, '종목명': str})[['종목명', '종목코드']]

names_lst = list(stock_code['종목명'].values)
days_lst = [5, 20, 60, 120]

for name in tqdm(names_lst):
    try:
        print(f'Start {name} Modeling!!')
        data, code = rr.load_stocks_data(f'{name}', stock_code)

        for day in days_lst:

            # Stocks 객체 생성
            stocks = rr.Stocks(data)

            # 모델링
            stocks.modeling(code, day)
    except:
        print(f'{name} ERROR!!')

print('Finish!!!!')