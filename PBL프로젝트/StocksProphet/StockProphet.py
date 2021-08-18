from tqdm import tqdm
import datetime
import warnings
import time
warnings.filterwarnings('ignore')
import itertools
import random
import pickle
from datetime import timedelta
import FinanceDataReader as fdr

import numpy as np
import pandas as pd
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
    diff_day = datetime.timedelta(days=365)
    
    start_date = str(today - diff_day)
    finish_date = str(today)
    
    try:
        data = fdr.DataReader(f'{code}', start_date, finish_date)
        time.sleep(1)
        print(data.shape)
        
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

    params_grid = {'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.5],
                   'seasonality_prior_scale': [1, 5, 10, 15],
                   'yearly_seasonality': [5, 10, 15, 20]
                   }

    grid = ParameterGrid(params_grid)
    cnt = 0

    for p in grid:
        cnt = cnt+1

    model_parameters = pd.DataFrame(columns = ['MAPE',
                                               'changepoint_prior_scale',
                                               'seasonality_prior_scale',
                                               'yearly_seasonality']
                                               )

    def __init__(self, data):
        self.data = data
    
    def modeling(self, code, day, grid=grid, model_parameters=model_parameters):
        
        train_data = self.data.iloc[:-day, :]
        test_data = self.data.iloc[-day:, :]
        test_data = test_data.set_index('ds')
        
        idx = 0
            
        for p in tqdm(grid):
            prophet = Prophet(yearly_seasonality=p['yearly_seasonality'],
                              weekly_seasonality=False,
                              daily_seasonality=False,
                              changepoint_prior_scale=p['changepoint_prior_scale'],
                              seasonality_prior_scale=p['seasonality_prior_scale']
                              )

            prophet.fit(train_data)

            future_data = prophet.make_future_dataframe(periods=day, freq='D')
            forecast_data = prophet.predict(future_data)

            pred_y = forecast_data.yhat.values[-day:]
            MAPE = mean_absolute_percentage_error(test_data.values, pred_y)
            
            if idx == 0:
                idx += 1
                model_parameters = model_parameters.append({'MAPE':MAPE,
                                                            'changepoint_prior_scale':p['changepoint_prior_scale'],
                                                            'seasonality_prior_scale':p['seasonality_prior_scale'],
                                                            'yearly_seasonality':p['yearly_seasonality']
                                                            },ignore_index=True)

                
            else:
                if MAPE < model_parameters['MAPE'].iloc[-1]:
                    model_parameters = model_parameters.append({'MAPE':MAPE,
                                                                'changepoint_prior_scale':p['changepoint_prior_scale'],
                                                                'seasonality_prior_scale':p['seasonality_prior_scale'],
                                                                'yearly_seasonality':p['yearly_seasonality']
                                                                },ignore_index=True)
                else:
                    continue

        parameter = model_parameters.iloc[-1, :]

        prophet = Prophet(yearly_seasonality=parameter['yearly_seasonality'],
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        changepoint_prior_scale=parameter['changepoint_prior_scale'],
                        seasonality_prior_scale=parameter['seasonality_prior_scale'])

        prophet.fit(self.data)

        with open(f'./model/prophet_{code}_{day}.pkl', "wb") as f:
            pickle.dump(prophet, f)

    def predict(self, code, day):

        with open(f'./model/prophet_{code}_{day}.pkl', 'rb') as f:
            prophet = pickle.load(f)

        future_data = prophet.make_future_dataframe(periods=day, freq='D')
        forecast_data = prophet.predict(future_data)

        pred = forecast_data.yhat.values[-day:]

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
        # return pred_day, pred