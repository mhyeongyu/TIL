import pandas as pd
import Stock as st
import joblib
from flask import Flask, request
from datetime import timedelta
# import jsonify

# company(기업명): ex) 삼성전자, SK하이닉스, LG화학, 카카오, NAVER, 현대차
# day(예측기간): ex) 5, 20, 60, 120

def hola_sec_function(name='삼성전자', day=5):

    stock_code = pd.read_csv('./KOSPI_200.csv', dtype={'종목코드': str, '종목명': str})[['종목명', '종목코드']]

    data, code = st.load_stocks_data(f'{name}', stock_code)
    stocks = st.Stocks(data, code, int(day))
    stocks.preprocessing()
    result = stocks.predict()

    return result

app = Flask(__name__)
# app.config['PERMANENT_SESSION_LIFETIME'] =  timedelta(minutes=5)

@app.route('/')
def info():
    input_company = request.args.get("company")
    input_day = request.args.get("day")

    msg = hola_sec_function(input_company, input_day)

    return str(msg)


if __name__ == '__main__':
    app.run(port=8888)