# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:09:44 2022

@author: HIDRAULICA-Dani
"""

import os
import datetime as dt
import pandas as pd
from urllib.request import urlretrieve
import shutil

os.chdir('D:/DANI/VARIOS/EMAS')
# os.listdir()

# Download new records
emas = ["MONTERREY", "CUMBRESDEMTYI", "CUMBRESDEMTYELDIENTE"]
today = dt.datetime.today()
today_str = str(today)[:4]+str(today)[5:7]+str(today)[8:10]

if not os.path.exists('./DATOS_NUEVOS/' + today_str):
    os.mkdir('./DATOS_NUEVOS/' + today_str)

for ema in emas:
    # https://smn.conagua.gob.mx/tools/PHP/sivea_v2/siveaEsri2/php/get_reporteEstacion.php?tipo=3&estacion=MONTERREY
    # tipo=3 es de ultimos 90 dias
    url = "https://smn.conagua.gob.mx/tools/PHP/sivea_v2/siveaEsri2/php/get_reporteEstacion.php?tipo=3&estacion="+ema
    urlretrieve(url, "./DATOS_NUEVOS/"+today_str+"/Estacion_"+ema+"__90_dias.csv")
    
# Add new records to complete files
# ALL VARIABLES
emas_full = os.listdir('./COMPLETAS')
emas_dates = os.listdir('./DATOS_NUEVOS')
emas_new = os.listdir('./DATOS_NUEVOS/'+emas_dates[-1])
vars = ['P', 'T', 'SR', 'RH', 'BP', 'WS', 'WSmax', 'WD', 'WDmax']
emas_var_names = {'Temperatura del Aire (°C)': 'T', 'Precipitación (mm)': 'P', 'Humedad relativa (%)': 'RH',
                  'Presión Atmosférica (hpa)': 'BP', 'Radiación Solar (W/m²)': 'SR', 'Dirección del Viento (grados)': 'WD',
                  'Rapidez de viento (km/h)': 'WS', 'Dirección de ráfaga (grados)': 'WDmax',
                  'Rapidez de ráfaga (km/h)': 'WSmax'}

for ema in emas_new:
    # ema = 'Estacion_CUMBRESDEMTYELDIENTE__90_dias.csv'
    print(ema)
    shutil.copyfile('./COMPLETAS/'+ema[9:-13]+'_All.csv', './OTRAS/'+ema[9:-13]+'_All_'+str(dt.datetime.today())[:10].replace('-','')+'.csv')
    emas_com = pd.read_csv('./COMPLETAS/'+ema[9:-13]+'_All.csv', parse_dates=['Unnamed: 0'], infer_datetime_format=True)
    emas_com.index = emas_com['Unnamed: 0']
    emas_com = emas_com.drop(['Unnamed: 0'], axis=1)
    emas_com.index.name = None

    try:
        emas = pd.read_csv('./DATOS_NUEVOS/'+emas_dates[-1]+'/Estacion_'+ema[9:-13]+'__90_dias.csv', skiprows=9, parse_dates=['Fecha UTC'], infer_datetime_format=True, encoding='latin-1')
    except:
        emas = pd.read_csv('./DATOS_NUEVOS/'+emas_dates[-1]+'/Estacion_'+ema[9:-13]+'__90_dias.csv', skiprows=10, parse_dates=['Fecha UTC'], infer_datetime_format=True, encoding='latin-1')        
    emas.index = emas['Fecha UTC']
    emas = emas.drop(['Fecha UTC', 'Fecha Local'], axis=1)
    emas.index.name = None
    emas = emas.rename(columns=emas_var_names)
    emas = emas[vars]
    emas = emas.sort_index()
    
    emas_up = pd.concat([emas_com, emas])
    emas_up = emas_up.dropna(how='all')
    emas_up = emas_up[~emas_up.index.duplicated(keep='first')]
    emas_up.to_csv('./COMPLETAS/'+ema[9:-13]+'_All.csv')