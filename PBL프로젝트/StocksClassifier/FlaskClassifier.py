import pandas as pd
import Classifier as cls
from flask import Flask, request

def hola_sec_function(name='삼성전자', day=5):

    # 종목명, 종목코드 불러오기
    stock_code = pd.read_csv('./KOSPI_200.csv', dtype={'종목코드': str, '종목명': str})[['종목명', '종목코드']]

    data, code = cls.load_stocks_data(f'{name}', stock_code) # 종목명, 종목코드로 주가데이터 로드
    stocks = cls.Stocks(data)                                # stocks 객체 생성
    stocks.preprocessing()
    sign_data = stocks.stocksign(stocks.data, int(day))      # stocks 객체안의 데이터 보조지표 생성 및 전처리
    result = stocks.predict(sign_data, code, int(day))       # 예측값 딕셔너리 형태로 반환

    return result

app = Flask(__name__)

# Input값, /?company=기업명&day=예측기간
@app.route('/')
def info():
    input_company = request.args.get("company") # company(기업명): ex) 삼성전자, SK하이닉스, LG화학, 카카오, NAVER, 현대차
    input_day = request.args.get("day")         # day(예측기간): ex) 5, 20, 60, 120

    msg = hola_sec_function(input_company, input_day)

    return str(msg)

if __name__ == '__main__':
    app.run(port=8000)