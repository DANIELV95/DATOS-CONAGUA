# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 11:47:03 2022

@author: HIDRAULICA-Dani
"""

import os
import pandas as pd
import datetime as dt
# import shutil
# from urllib.request import urlretrieve
import requests
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
disable_warnings(InsecureRequestWarning)

os.chdir('D:/DANI/VARIOS/EMAS')
#Read emas names
emas = pd.read_csv('data_emas.csv')
emas = emas.drop(['Unnamed: 0'], axis=1)
emas_names = emas['estacion_m'].values #Estaciones a actualizar
# emas_names = ['MONTERREY', 'CUMBRESDEMTYI', 'CUMBRESDEMTYELDIENTE'] #Estaciones a actualizar
vars = ['P', 'T', 'SR', 'RH', 'BP', 'WS', 'WSmax', 'WD', 'WDmax']
emas_var_names = {'Temperatura del Aire (°C)': 'T', 'Precipitación (mm)': 'P', 'Humedad relativa (%)': 'RH',
                  'Presión Atmosférica (hpa)': 'BP', 'Radiación Solar (W/m²)': 'SR', 'Dirección del Viento (grados)': 'WD',
                  'Rapidez de viento (km/h)': 'WS', 'Dirección de ráfaga (grados)': 'WDmax',
                  'Rapidez de ráfaga (km/h)': 'WSmax'}
today = dt.datetime.today()
today_str = today.strftime('%Y%m%d')

if not os.path.exists('./DATOS_NUEVOS/' + today_str):
    os.mkdir('./DATOS_NUEVOS/' + today_str)

for ema in emas_names:
    print(ema)
    # ema = 'ACAPULCO'
    # https://smn.conagua.gob.mx/tools/PHP/sivea_v2/siveaEsri2/php/get_reporteEstacion.php?tipo=3&estacion=MONTERREY
    # tipo=3 es de ultimos 90 dias
    # urlretrieve(url, "./DATOS_NUEVOS/"+today_str+"/Estacion_"+ema+"__90_dias.csv")
    url = "https://smn.conagua.gob.mx/tools/PHP/sivea_v2/siveaEsri2/php/get_reporteEstacion.php?tipo=3&estacion="+ema
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36',
               'Referer': 'https://smn.conagua.gob.mx/tools/PHP/sivea_v2/sivea.php'}
    response = requests.get(url, headers=headers, verify=False)
    f = open("./DATOS_NUEVOS/"+today_str+"/Estacion_"+ema+"__90_dias.csv", "wb")
    f.write(response.content)
    f.close()
   
for ema in emas_names:
    if not os.path.exists('./COMPLETAS/' + ema + '_All.csv'):
        f = pd.DataFrame([], columns=vars)
        f.to_csv('./COMPLETAS/' + ema + '_All.csv') #, encoding='latin-1')

emas_full = os.listdir('./COMPLETAS')
emas_dates = os.listdir('./DATOS_NUEVOS')
emas_new = os.listdir('./DATOS_NUEVOS/'+emas_dates[-1])

for ema in emas_new:
    # ema = 'Estacion_CUMBRESDEMTYELDIENTE__90_dias.csv'
    print(ema)
    # shutil.copyfile('./COMPLETAS/'+ema[9:-13]+'_All.csv', './OTRAS/'+ema[9:-13]+'_All_'+today.strftime('%Y%m%d')+'.csv')
    try:
        emas_com = pd.read_csv('./COMPLETAS/'+ema[9:-13]+'_All.csv', parse_dates=['Unnamed: 0'], infer_datetime_format=True)
        emas_com.index = emas_com['Unnamed: 0']
        emas_com = emas_com.drop(['Unnamed: 0'], axis=1)
        emas_com.index.name = None
    except Exception:
        pass

    try:
        emas = pd.read_csv('./DATOS_NUEVOS/'+emas_dates[-1]+'/Estacion_'+ema[9:-13]+'__90_dias.csv', skiprows=8, parse_dates=['Fecha UTC'], infer_datetime_format=True, encoding='latin-1')
        emas.index = emas['Fecha UTC']
        emas = emas.drop(['Fecha UTC', 'Fecha Local'], axis=1)
        emas.index.name = None
        emas = emas.rename(columns=emas_var_names)
        emas = emas.loc[:, ~emas.columns.str.contains('^Unnamed')]
        # emas = emas[vars]
        emas = emas.sort_index()
        
        emas_up = pd.concat([emas_com, emas])
        emas_up = emas_up.dropna(how='all')
        emas_up = emas_up[~emas_up.index.duplicated(keep='first')]
        emas_up.to_csv('./COMPLETAS/'+ema[9:-13]+'_All.csv')
        
    except Exception:
        print(ema)
        pass
        # emas = pd.read_csv('./DATOS_NUEVOS/'+emas_dates[-1]+'/Estacion_'+ema[9:-13]+'__90_dias.csv', skiprows=10, parse_dates=['Fecha UTC'], infer_datetime_format=True, encoding='latin-1')
    