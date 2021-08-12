import pandas as pd
import Stock as st
import joblib
from flask import Flask, request
from datetime import timedelta

# company(기업명): ex) 삼성전자, SK하이닉스, LG화학, 카카오, NAVER, 현대차
# day(예측기간): ex) 5, 20, 60, 120

def hola_sec_function(name='삼성전자', day=5):
    print('start')
    stock_code = pd.read_csv('./KOSPI_200.csv', dtype={'종목코드': str, '종목명': str})[['종목명', '종목코드']]

    data, code = st.load_stocks_data(f'{name}', stock_code)
    stock = st.Stocks(data, code, int(day))
    stock.preprocessing()

    result = stock.predict()

    return result

app = Flask(__name__)

@app.route('/')
def info():
    msg = hola_sec_function('삼성전자', 5)

    return str(msg)

if __name__ == '__main__':
    app.run(port=8888)
