from tqdm import tqdm
import datetime
import warnings
import time
warnings.filterwarnings('ignore')
import itertools
import random
import pickle
from datetime import timedelta

import numpy as np
import pandas as pd
import pandas_datareader as pdr
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_columns', None)
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterGrid

from fbprophet import Prophet

# stock_code = pd.read_csv('../3.프로젝트/Data/KOSPI_200.csv', dtype={'종목코드': str, '종목명': str})[['종목명', '종목코드']]

def load_stocks_data(name, stock_code):
    
    codes_dic = dict(stock_code.values)
    code = codes_dic[name]

    today = datetime.date.today()
    diff_day = datetime.timedelta(days=10000)
    
    start_date = '2020-02-01'
    finish_date = str(today)
    
    try:
        data = pdr.DataReader(f'{code}.KS','yahoo', start_date, finish_date)
        time.sleep(1)
        print(data.shape)
        
        data = data[data['Volume'] != 0]
        data = data.reset_index()
        data = data[['Date', 'Close']]
        data.columns = ['ds', 'y']
        return data, code

    except:
        print(f'     LOAD ERROR: {name}     ')

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class Stocks:

    params_grid = {'changepoint_prior_scale':[0.1,0.01,0.001],
                'n_changepoints' : [50,100,150],
                'fourier_order' : [5,10],
                'period':[0.1,0.3,0.5]
                }
    grid = ParameterGrid(params_grid)
    cnt = 0

    for p in grid:
        cnt = cnt+1

    model_parameters = pd.DataFrame(columns = ['MAPE',
                                               'changepoint_prior_scale',
                                               'n_changepoints',
                                               'fourier_order',
                                               'period'])

    def __init__(self, data):
        self.data = data
    
    def modeling(self, code, day, grid=grid, model_parameters=model_parameters):
        
        train_data = self.data.iloc[:-day, :]
        test_data = self.data.iloc[-day:, :]
        test_data = test_data.set_index('ds')
        
        idx = 0
            
        for p in grid:
            prophet = Prophet(seasonality_mode='multiplicative', 
                            yearly_seasonality=False,
                            weekly_seasonality=False,
                            daily_seasonality=False,
                            changepoint_prior_scale=p['changepoint_prior_scale'],
                            n_changepoints=p['n_changepoints'],
                            )
            prophet.add_seasonality(name='seasonality_1',period=p['period'],fourier_order=p['fourier_order'])
            prophet.fit(train_data)

            future_data = prophet.make_future_dataframe(periods=day, freq='min')
            forecast_data = prophet.predict(future_data)

            pred_y = forecast_data.yhat.values[-day:]
            MAPE = mean_absolute_percentage_error(test_data.values, pred_y)
            
            if idx == 0:
                idx += 1
                model_parameters = model_parameters.append({'MAPE':MAPE,
                                                            'changepoint_prior_scale':p['changepoint_prior_scale'],
                                                            'n_changepoints':p['n_changepoints'],
                                                            'fourier_order':p['fourier_order'],
                                                            'period':p['period']
                                                            },ignore_index=True)

                
            else:
                if MAPE < model_parameters['MAPE'].iloc[-1]:
                    model_parameters = model_parameters.append({'MAPE':MAPE,
                                                    'changepoint_prior_scale':p['changepoint_prior_scale'],
                                                    'n_changepoints':p['n_changepoints'],
                                                    'fourier_order':p['fourier_order'],
                                                    'period':p['period']
                                                    },ignore_index=True)
                else:
                    continue

        parameter = model_parameters.iloc[-1, :]

        prophet = Prophet(seasonality_mode='multiplicative', 
                            yearly_seasonality=False,
                            weekly_seasonality=False,
                            daily_seasonality=False,
                            changepoint_prior_scale=parameter['changepoint_prior_scale'],
                            n_changepoints=int(parameter['n_changepoints']),
                            )

        prophet.add_seasonality(name='seasonality_1', period=0.1, fourier_order=int(parameter['fourier_order']))
        prophet.add_seasonality(name='seasonality_2', period=0.3, fourier_order=int(parameter['fourier_order']))
        prophet.add_seasonality(name='seasonality_3', period=0.5, fourier_order=int(parameter['fourier_order']))

        prophet.fit(self.data)

        with open(f'./model/prophet_{code}_{day}.pkl', "wb") as f:
            pickle.dump(prophet, f)

    def predict(self, code, day):

        with open(f'./model/prophet_{code}_{day}.pkl', 'rb') as f:
            prophet = pickle.load(f)

        future_data = prophet.make_future_dataframe(periods=day, freq='min')
        forecast_data = prophet.predict(future_data)

        pred = forecast_data.yhat.values[-20:]

        week = day//5

        first_day = forecast_data.iloc[-day*2, 0] + timedelta(weeks=week)
        finish_day = forecast_data.iloc[-day-1, 0] + timedelta(weeks=week)
        
        day_range = pd.date_range(first_day, finish_day)
        pred_day = np.array(day_range[day_range.dayofweek < 5].strftime('%Y-%m-%d'))

        # dictonary생성, (key:날짜 value:예측값)
        result_dic = {}
        for i, j in zip(pred_day, pred):
            result_dic[i] = j

        return result_dic