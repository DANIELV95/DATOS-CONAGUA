# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:20:14 2022

@author: HIDRAULICA-Dani
"""

import pandas as pd
import numpy as np
import os
from bs4 import BeautifulSoup
import glob
import shutil
import dateparser as dp
import datetime as dt
import xlrd
from xlrd import XLRDError
# import win32com.client as win32
import warnings
import sys

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


os.chdir('D:/DANI/VARIOS/EMAS/DATOS_ORIGINALES/TODOS')
years = os.listdir()

#Convert XLS file to XLSX
def convert_to_xlsx(fname_xls, fname_xlsx):
    with open(fname_xls) as xml_file:
        soup = BeautifulSoup(xml_file.read(), 'xml')
        writer = pd.ExcelWriter(fname_xlsx)
        for sheet in soup.findAll('Worksheet'):
            sheet_as_list = []
            for row in sheet.findAll('Row'):
                sheet_as_list.append([cell.Data.text if cell.Data else '' for cell in row.findAll('Cell')])
            pd.DataFrame(sheet_as_list).to_excel(writer, sheet_name=sheet.attrs['ss:Name'], index=False, header=False)
        writer.save()

#Get unique headers for all files
headers = pd.DataFrame()
i = -1
for year in years:
    # year = '2019'
    emas_in_year = os.listdir('./'+year)
    for ema in emas_in_year:
        i += 1
        # ema = 'ACAPONETA_2004_10.xls'
        fname_xls = './'+year+'/'+ema
        fname_xlsx = fname_xls+'x'
        try:
            df = pd.read_excel(fname_xls)
        except ValueError:
            df = pd.read_csv(fname_xls, encoding= 'unicode_escape')
        except:
            convert_to_xlsx(fname_xls, fname_xlsx)
            df = pd.read_excel(fname_xlsx)
        print(year, i, ema)
        headers.loc[:,i] = pd.concat([pd.DataFrame([fname_xls[7:-4]], ['id']), pd.Series(np.array(df.columns))], axis=0)
headers = headers.T
stations = headers.id
headers = headers.drop(['id'], axis=1).drop_duplicates()
diff_stations = headers.join(stations)
# stations.to_csv('stations.csv')
# headers.to_csv('headers.csv')
# diff_stations.to_csv('diff_stations.csv')

#Move file with errors while read_excel
for year in years:
    # year = '2004'
    for fname_xlsx in glob.glob('.\\'+year+'\*.xlsx'):
        fname_xls = fname_xlsx[:-1]
        shutil.move('./'+year+'/'+fname_xls[7:], '../XLS_MOD/'+fname_xls[7:])
        print(year, fname_xlsx)



dif = pd.read_csv('../diff_stations.csv', encoding= 'unicode_escape')
dif = dif.drop(['Unnamed: 0', 'id'], axis=1)
difm = pd.read_csv('../diff_stations_mod.csv', encoding= 'unicode_escape')
difm = difm.drop(['i', 'id'], axis=1)

#Arrange EMAS data in CSV file with same format
# f = open('../ERRORS.txt', 'w')
for year in years:
    # year = '2014'
    emas_in_year = os.listdir('./'+year)
    for ema in emas_in_year:
        # i += 1
        print(ema)
        # ema = 'SANTA ROSALIA_2004_07.xls'
        # ema = 'ACAPULCO_2014.csv'
        # ema = emas_in_year[0]
        try:
            try:
                fname_xls = ema
                workbook = xlrd.open_workbook('./'+year+'/'+fname_xls, logfile=open(os.devnull, "w"))
                df = pd.read_excel(workbook)
                fname_csv_final = fname_xls[:-4]+'.csv'
            except XLRDError:
                try:
                    fname_xls = ema
                    df = pd.read_excel('./'+year+'/'+fname_xls)
                    fname_csv_final = fname_xls[:-4]+'.csv'
                except ValueError:
                    fname_csv = ema
                    df = pd.read_csv('./'+year+'/'+fname_csv, encoding= 'unicode_escape')
                    fname_csv_final = ema
            except:
                fname_xlsx = ema+'x'
                workbook = xlrd.open_workbook('./'+year+'/'+fname_xlsx, logfile=open(os.devnull, "w"))
                df = pd.read_excel(workbook)
                fname_xlsx[:-5]+'.csv'
            
            a = np.empty(26-len(df.columns.values))
            a[:] = np.nan
            cols = pd.DataFrame(np.concatenate([df.columns.values,a])).T
            
            for i, row in dif.iterrows():
                # i = 22
                # row = dif.loc[i]
                x = dif.loc[i].dropna().values
                y = cols.loc[0].dropna().values.astype(str)
                check = np.array_equal(x,y)
                # print(i, check)
                if check == True:
                    df.columns = difm.loc[i].dropna().values
                    col = list(df.columns.values)
                    for x in ['Date', 'Time', 'Datetime']:
                        if x in col:
                            col.remove(x)
                    try:
                        for c in col:
                            df[c] = pd.to_numeric(df[c], errors='coerce')
                    except:
                        pass
                    df[col] = df[col].applymap(lambda x: np.nan if isinstance(x, str) else x)
                    df = df.replace(r'^\s*$', np.nan, regex=True)
                    df = df.replace(r'^([A-Za-z])+$', np.nan, regex=True)
                    df = df.dropna(how='all')
            
            if 'Date' in df.columns:
                df['Datetime'] = np.nan
                try:
                    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.time
                except:
                    pass
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                except:
                    for i, row in df.iterrows():
                        # print(row)
                        df['Date'][i] = dp.parse(df['Date'][i], languages=['es'])
                for i, row in df.iterrows():
                    df['Datetime'][i] = dt.datetime.combine(df['Date'][i], df['Time'][i])
                df.index = df['Datetime']
                df = df.drop(['Date', 'Time', 'Datetime'], axis=1)
                df.index.name = None
            else:
                if df['Datetime'].dtype != '<M8[ns]':
                    for i, row in df.iterrows():
                        # print(row)
                        df['Datetime'][i] = dp.parse(df['Datetime'][i], languages=['es'])
                    df.index = df['Datetime']
                    df = df.drop(['Datetime'], axis=1)
                    df.index.name = None
                else:
                    df.index = df['Datetime']
                    df = df.drop(['Datetime'], axis=1)
                    df.index.name = None
                    # print('Datetime Format OK')
            
            df = df.sort_index()
            df.to_csv('../CSV/'+fname_csv_final)
            
        except:
            print('ERROR_'+year+'_'+ema, file=f)
f.close()



dif = pd.read_csv('../diff_stations.csv', encoding= 'unicode_escape')
dif = dif.drop(['Unnamed: 0', 'id'], axis=1)
difm = pd.read_csv('../diff_stations_mod.csv', encoding= 'unicode_escape')
difm = difm.drop(['i', 'id'], axis=1)

#Add files with errors
f = open('../ERRORS.txt', 'r')
errors = f.readlines()
# errors.replace('\n', '')
errors = list(map(lambda x: x.replace('\n', ''), errors))
errors = list(map(lambda x: x.replace('ERROR_', ''), errors))
# emas = list(map(lambda x: x[5:], errors))
# years = list(map(lambda x: x[:4], errors))
# errors = list(map(lambda x: x.rsplit( '.', 1)[0], errors))
# errors[0][5:]
# errors = list(map(lambda x: x.replace('_', ''), errors))
# for i in range(10):
#     print(i)
#     errors = list(map(lambda x: x.replace(str(i), ''), errors))

for error in errors:
    print(error)
    # ema = errors[0][5:]
    # year = errors[0][:4]
    ema = error[5:]
    year = error[:4]
    
    try:
        try:
            fname_xls = ema
            workbook = xlrd.open_workbook('./'+year+'/'+fname_xls, logfile=open(os.devnull, "w"))
            df = pd.read_excel(workbook)
            fname_csv_final = fname_xls[:-4]+'.csv'
        except XLRDError:
            try:
                fname_xls = ema
                df = pd.read_excel('./'+year+'/'+fname_xls)
                fname_csv_final = fname_xls[:-4]+'.csv'
            except ValueError:
                fname_csv = ema
                df = pd.read_csv('./'+year+'/'+fname_csv, encoding= 'unicode_escape')
                fname_csv_final = ema
        except:
            fname_xlsx = ema+'x'
            workbook = xlrd.open_workbook('./'+year+'/'+fname_xlsx, logfile=open(os.devnull, "w"))
            df = pd.read_excel(workbook)
            fname_xlsx[:-5]+'.csv'
        
        a = np.empty(26-len(df.columns.values))
        a[:] = np.nan
        cols = pd.DataFrame(np.concatenate([df.columns.values,a])).T
        
        for i, row in dif.iterrows():
            # i = 22
            # row = dif.loc[i]
            x = dif.loc[i].dropna().values
            y = cols.loc[0].dropna().values.astype(str)
            check = np.array_equal(x,y)
            print(i, check)
            if check == True:
                df.columns = difm.loc[i].dropna().values
                col = list(df.columns.values)
                for x in ['Date', 'Time', 'Datetime']:
                    if x in col:
                        col.remove(x)
                try:
                    for c in col:
                        df[c] = pd.to_numeric(df[c], errors='coerce')
                except:
                    pass
                df[col] = df[col].applymap(lambda x: np.nan if isinstance(x, str) else x)
                df = df.replace(r'^\s*$', np.nan, regex=True)
                df = df.replace(r'^([A-Za-z])+$', np.nan, regex=True)
                # df = df.replace(':', np.nan, regex=True)
                df = df.dropna(how='all')
        
        if 'Date' in df.columns:
            df['Datetime'] = np.nan
            try:
                df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.time
            except ValueError:
                for i, row in df.iterrows():
                    # print(i, row)
                    try:
                        df['Time'].loc[i] = (dt.datetime(2010,1,1) + pd.Timedelta(df['Time'].loc[i]*1440, unit='m')).time()
                    except:
                        pass
            except:
                df['Time'] = pd.to_datetime(dt.datetime(2010,1,1) + pd.TimedeltaIndex(df['Time']*1440, unit='m')).time
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except:
                for i, row in df.iterrows():
                    # print(row)
                    df['Date'][i] = dp.parse(df['Date'][i], languages=['es'])
            for i, row in df.iterrows():
                df['Datetime'][i] = dt.datetime.combine(df['Date'][i], df['Time'][i])
            df.index = df['Datetime']
            df = df.drop(['Date', 'Time', 'Datetime'], axis=1)
            df.index.name = None
        else:
            if df['Datetime'].dtype != '<M8[ns]':
                try:
                    for i, row in df.iterrows():
                        # print(row)
                        df['Datetime'][i] = dp.parse(df['Datetime'][i], languages=['es'])
                    df.index = df['Datetime']
                    df = df.drop(['Datetime'], axis=1)
                    df.index.name = None
                except:
                    df.index = df['Datetime']
                    df = df.drop(['Datetime'], axis=1)
                    df.index.name = None
            else:
                df.index = df['Datetime']
                df = df.drop(['Datetime'], axis=1)
                df.index.name = None
                # print('Datetime Format OK')
        
        df = df.sort_index()
        df = df.dropna(how='all')
        df.to_csv('../CSV/'+fname_csv_final)

    except:
        print('ERROR', ema, year)
        pass

#Check names
emas_files = os.listdir('../CSV/')
for ema_file in emas_files:
    try:
        ema = ema_file.replace('..','.')
        if ema != ema_file:
            if os.path.exists('../CSV/'+ema):
                os.remove('../CSV/'+ema)
            os.rename('../CSV/'+ema_file, '../CSV/'+ema)
            print(ema_file)
    except:
        pass

#Get unique EMAS names in same format
emas = []
emas_files = os.listdir('../CSV/')

for ema_file in emas_files:
    print(ema_file)
    # ema = 'CANCUN_2013_09.csv'
    ema = ema_file[:-4]
    if ema[-4:].isnumeric():
        ema = ema[:-5]
    elif ema[-2:].isnumeric():
        ema = ema[:-8]
    else:
        pass
    try:
        ema = ema.replace('-',' ')
        # ema = ema.replace('_','')
    except:
        pass
    if ema not in emas:
        emas.append(ema)
emas = pd.DataFrame(emas)
# emas.to_csv('../names_emas.csv', encoding='latin1')

#Merge all files in one for each station
emas = ['']
emas_files = os.listdir('../CSV/')
emas_files.remove('COMPLETAS')
ema_complete = pd.DataFrame()

for ema_file in emas_files:
    # print(ema_file)
    # ema = 'CANCUN_2013_09.csv'
    # ema_file = emas_files[0]
    df = pd.read_csv('../CSV/'+ema_file, parse_dates=['Unnamed: 0'])
    df.index = df['Unnamed: 0']
    df = df.drop(['Unnamed: 0'], axis=1)
    df.index.name = None
    df = df.dropna(how='all')
    
    ema = ema_file[:-4]
    if ema[-4:].isnumeric():
        ema = ema[:-5]
    elif ema[-2:].isnumeric():
        ema = ema[:-8]
    else:
        pass
    try:
        ema = ema.replace('-',' ')
        # ema = ema.replace('_','')
    except:
        pass
    # print(ema)
    if ema == emas[-1]:
        # print('append file')
        ema_complete = pd.concat([ema_complete, df])
        emas.append(ema)
    else:
        if ema not in emas:
            print('Write file '+emas[-1])
            ema_complete.to_csv('../CSV/COMPLETAS/'+emas[-1]+'.csv')
            ema_complete = pd.DataFrame()
            
            print(ema)
            ema_complete = pd.concat([ema_complete, df])
            emas.append(ema)
os.delete('../CSV/COMPLETAS/.csv')

#Combine similar stations with different names
names = pd.read_csv('../data_emas_up.csv', index_col='Unnamed: 0', encoding='latin1')
secon_names = names['name_files_2'].dropna()
main_names = names['name_files'][secon_names.index]

for ema_id in secon_names.index:
    # ema_id = 192
    ema = main_names[ema_id]
    df = pd.read_csv('../CSV/COMPLETAS/'+ema+'.csv', parse_dates=['Unnamed: 0'])
    df.index = df['Unnamed: 0']
    df = df.drop(['Unnamed: 0'], axis=1)
    df.index.name = None
    df = df.dropna(how='all')
    
    ema_secon = secon_names[ema_id]
    df_secon = pd.read_csv('../CSV/COMPLETAS/'+ema_secon+'.csv', parse_dates=['Unnamed: 0'])
    df_secon.index = df_secon['Unnamed: 0']
    df_secon = df_secon.drop(['Unnamed: 0'], axis=1)
    df_secon.index.name = None
    df_secon = df_secon.dropna(how='all')
    
    print(ema)
    os.rename('../CSV/COMPLETAS/'+ema+'.csv', '../CSV/DIFERENTES/'+ema+'.csv')
    os.rename('../CSV/COMPLETAS/'+ema_secon+'.csv', '../CSV/DIFERENTES/'+ema_secon+'.csv')
    ema_complete = pd.concat([df, df_secon])
    ema_complete = ema_complete.sort_index()
    ema_complete = ema_complete.drop_duplicates()
    ema_complete = ema_complete[~ema_complete.index.duplicated(keep='first')]
    ema_complete.to_csv('../CSV/COMPLETAS/'+ema+'.csv')

#Identify emas with datetime errors
emas_files = os.listdir('../CSV/COMPLETAS')
f = open('../DATETIME_ERRORS.txt', 'w')

for ema in emas_files:
    print(ema)
    # ema = emas_files[0]
    # ema = 'JALAPA.csv'
    df = pd.read_csv('../CSV/COMPLETAS/'+ema, parse_dates=['Unnamed: 0'])
    df.index = df['Unnamed: 0']
    df = df.drop(['Unnamed: 0'], axis=1)
    df.index.name = None
    df = df.dropna(how='all')
    df = df.sort_index()
    if pd.isnull(df.index[-1]):
        print('DATETIME ERROR '+ema, file=f)
f.close()


#Identify emas with datetime errors same days
emas_files = os.listdir('../CSV/COMPLETAS')
f = open('../DATETIME_SAME_ERRORS.txt', 'w')

for ema in emas_files:
    print(ema)
    # ema = emas_files[0]
    # ema = 'ACAPULCO.csv'
    df = pd.read_csv('../CSV/COMPLETAS/'+ema, parse_dates=['Unnamed: 0'])
    df.index = df['Unnamed: 0']
    df = df.drop(['Unnamed: 0'], axis=1)
    df.index.name = None
    df = df.dropna(how='all')
    df = df.sort_index()
    duplicates = len(df) - len(df.index.drop_duplicates())
    if duplicates > 10:
        print('DATETIME ERROR '+ema+' '+str(duplicates), file=f)
f.close()


#Identify emas indivual files with datetime errors
f = open('../DATETIME_ERRORS.txt', 'r')
errors = f.readlines()
errors = list(map(lambda x: x.replace('\n', ''), errors))
errors = list(map(lambda x: x.replace('DATETIME ERROR ', ''), errors))

f = open('../DATETIME_ERRORS_IND.txt', 'w')

# emas_files = os.listdir('../CSV/COMPLETAS')
for error in errors:
    print(error)
    # error = errors[0]
    # ema = error[:-4]
    ema_name = error[:-4]
    emas_files = [file for file in os.listdir('../CSV/') if file.startswith(ema_name)]

    for ema in emas_files:
        print(ema)
        # ema = emas_files[0]
        # ema = 'JALAPA.csv'
        df = pd.read_csv('../CSV/'+ema, parse_dates=['Unnamed: 0'])
        df.index = df['Unnamed: 0']
        df = df.drop(['Unnamed: 0'], axis=1)
        df.index.name = None
        df = df.dropna(how='all')
        df = df.sort_index()
        if pd.isnull(df.index[-1]):
            print('DATETIME ERROR '+ema, file=f)
f.close()


#Identify emas with datetime errors same days individual
f = open('../DATETIME_SAME_ERRORS.txt', 'r')
errors = f.readlines()
errors = list(map(lambda x: x.replace('\n', ''), errors))
errors = list(map(lambda x: x.replace('DATETIME ERROR ', ''), errors))

f = open('../DATETIME_SAME_ERRORS_IND.txt', 'w')

for error in errors:
    print(error)
    # error = errors[0]
    # ema = error[:-4]
    ema_name = error.split('.csv', 1)[0]
    emas_files = [file for file in os.listdir('../CSV/') if file.startswith(ema_name)]

    for ema in emas_files:
        print(ema)
        # ema = emas_files[0]
        # ema = 'ACAPONETA_2004_01.csv'
        df = pd.read_csv('../CSV/'+ema, parse_dates=['Unnamed: 0'])
        df.index = df['Unnamed: 0']
        df = df.drop(['Unnamed: 0'], axis=1)
        df.index.name = None
        df = df.dropna(how='all')
        df = df.sort_index()
        duplicates = len(df) - len(df.index.drop_duplicates())
        if duplicates > 10:
            print('DATETIME ERROR '+ema+' '+str(duplicates), file=f)
f.close()


#Add files with datetime errors
f = open('../DATETIME_ERRORS.txt', 'r')
errors = f.readlines()
errors = list(map(lambda x: x.replace('\n', ''), errors))
errors = list(map(lambda x: x.replace('DATETIME ERROR ', ''), errors))

dif = pd.read_csv('../diff_stations.csv', encoding= 'unicode_escape')
dif = dif.drop(['Unnamed: 0', 'id'], axis=1)
difm = pd.read_csv('../diff_stations_mod.csv', encoding= 'unicode_escape')
difm = difm.drop(['i', 'id'], axis=1)

for error in errors:
    print(error)
    # error = errors[0]
    # ema = error[:-4]
    ema = error[:-4]
    
    for year in years:
        # year = '2004'
        files_in_year = [file for file in os.listdir('./'+year) if file.startswith(ema)]
        
        for file in files_in_year:
            # i += 1
            # print(file)
            # file = files_in_year[8]
    
            try:
                try:
                    fname_xls = file
                    workbook = xlrd.open_workbook('./'+year+'/'+fname_xls, logfile=open(os.devnull, "w"))
                    df = pd.read_excel(workbook)
                    fname_csv_final = fname_xls[:-4]+'.csv'
                except XLRDError:
                    try:
                        fname_xls = file
                        df = pd.read_excel('./'+year+'/'+fname_xls)
                        fname_csv_final = fname_xls[:-4]+'.csv'
                    except ValueError:
                        fname_csv = file
                        df = pd.read_csv('./'+year+'/'+fname_csv, encoding= 'unicode_escape')
                        fname_csv_final = file
                except:
                    fname_xlsx = file+'x'
                    workbook = xlrd.open_workbook('./'+year+'/'+fname_xlsx, logfile=open(os.devnull, "w"))
                    df = pd.read_excel(workbook)
                    fname_xlsx[:-5]+'.csv'
                
                a = np.empty(26-len(df.columns.values))
                a[:] = np.nan
                cols = pd.DataFrame(np.concatenate([df.columns.values,a])).T
                
                for i, row in dif.iterrows():
                    # i = 22
                    # row = dif.loc[i]
                    x = dif.loc[i].dropna().values
                    y = cols.loc[0].dropna().values.astype(str)
                    check = np.array_equal(x,y)
                    # print(i, check)
                    if check == True:
                        df.columns = difm.loc[i].dropna().values
                        col = list(df.columns.values)
                        for x in ['Date', 'Time', 'Datetime']:
                            if x in col:
                                col.remove(x)
                        try:
                            for c in col:
                                df[c] = pd.to_numeric(df[c], errors='coerce')
                        except:
                            pass
                        df[col] = df[col].applymap(lambda x: np.nan if isinstance(x, str) else x)
                        df = df.replace(r'^\s*$', np.nan, regex=True)
                        df = df.replace(r'^([A-Za-z])+$', np.nan, regex=True)
                        # df = df.replace(':', np.nan, regex=True)
                        df = df.dropna(how='all')
                
                if 'Date' in df.columns:
                    df['Datetime'] = np.nan
                    try:
                        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.time
                    except ValueError:
                        for i, row in df.iterrows():
                            # print(i, row)
                            try:
                                df['Time'].loc[i] = (dt.datetime(2010,1,1) + pd.Timedelta(df['Time'].loc[i]*1440, unit='m')).time()
                            except:
                                pass
                    except:
                        df['Time'] = pd.to_datetime(dt.datetime(2010,1,1) + pd.TimedeltaIndex(df['Time']*1440, unit='m')).time
                    try:
                        df['Date'] = pd.to_datetime(df['Date'])
                    except:
                        for i, row in df.iterrows():
                            # print(row)
                            df['Date'][i] = dp.parse(df['Date'][i], languages=['es'])
                    for i, row in df.iterrows():
                        df['Datetime'][i] = dt.datetime.combine(df['Date'][i], df['Time'][i])
                    df.index = df['Datetime']
                    df = df.drop(['Date', 'Time', 'Datetime'], axis=1)
                    df.index.name = None
                else:
                    if df['Datetime'].dtype != '<M8[ns]':
                        try:
                            df['Datetime'] = df['Datetime'].str.replace(' a.','')
                            df['Datetime'] = df['Datetime'].str.replace(' p.','')
                            # df['Datetime'] = df['Datetime'].str.replace('.',':')
                            df['Datetime'] = pd.to_datetime(df['Datetime'], format='%m/%d/%Y %H.%M.%S')
                        except:
                            pass
                        try:
                            for i, row in df.iterrows():
                                # print(row)
                                df['Datetime'][i] = dp.parse(df['Datetime'][i], languages=['es'])
                            df.index = df['Datetime']
                            df = df.drop(['Datetime'], axis=1)
                            df.index.name = None
                        except:
                            df.index = df['Datetime']
                            df = df.drop(['Datetime'], axis=1)
                            df.index.name = None
                    else:
                        df.index = df['Datetime']
                        df = df.drop(['Datetime'], axis=1)
                        df.index.name = None
                        # print('Datetime Format OK')
            
                df = df.sort_index()
                df = df.dropna(how='all')
                df.to_csv('../CSV/'+fname_csv_final)
    
            except:
                print('ERROR', ema, year)
                pass
    

#Add files with datetime errors individual
f = open('../DATETIME_ERRORS_IND.txt', 'r')
errors = f.readlines()
errors = list(map(lambda x: x.replace('\n', ''), errors))
errors = list(map(lambda x: x.replace('DATETIME ERROR ', ''), errors))

dif = pd.read_csv('../diff_stations.csv', encoding= 'unicode_escape')
dif = dif.drop(['Unnamed: 0', 'id'], axis=1)
difm = pd.read_csv('../diff_stations_mod.csv', encoding= 'unicode_escape')
difm = difm.drop(['i', 'id'], axis=1)

for error in errors:
    print(error)
    # error = errors[0]
    # ema = error[:-4]
    ema = error[:-4]
    if ema[-4:].isnumeric():
        year = ema[-4:]
        ema = ema[:-5]
        file = [file for file in os.listdir('./'+year) if file.startswith(ema+'_'+year)]
    elif ema[-2:].isnumeric():
        month = ema[-2:]
        year = ema[-7:-3]
        ema = ema[:-8]
        file = [file for file in os.listdir('./'+year) if file.startswith(ema+'_'+year+'_'+month)]
    else:
        pass
    
    file = file[0]
    
    try:
        try:
            fname_xls = file
            workbook = xlrd.open_workbook('./'+year+'/'+fname_xls, logfile=open(os.devnull, "w"))
            df = pd.read_excel(workbook)
            fname_csv_final = fname_xls[:-4]+'.csv'
        except XLRDError:
            try:
                fname_xls = file
                df = pd.read_excel('./'+year+'/'+fname_xls)
                fname_csv_final = fname_xls[:-4]+'.csv'
            except ValueError:
                fname_csv = file
                df = pd.read_csv('./'+year+'/'+fname_csv, encoding= 'unicode_escape')
                fname_csv_final = file
        except:
            fname_xlsx = file+'x'
            workbook = xlrd.open_workbook('./'+year+'/'+fname_xlsx, logfile=open(os.devnull, "w"))
            df = pd.read_excel(workbook)
            fname_xlsx[:-5]+'.csv'
        
        a = np.empty(26-len(df.columns.values))
        a[:] = np.nan
        cols = pd.DataFrame(np.concatenate([df.columns.values,a])).T
        
        for i, row in dif.iterrows():
            # i = 22
            # row = dif.loc[i]
            x = dif.loc[i].dropna().values
            y = cols.loc[0].dropna().values.astype(str)
            check = np.array_equal(x,y)
            # print(i, check)
            if check == True:
                df.columns = difm.loc[i].dropna().values
                col = list(df.columns.values)
                for x in ['Date', 'Time', 'Datetime']:
                    if x in col:
                        col.remove(x)
                try:
                    for c in col:
                        df[c] = pd.to_numeric(df[c], errors='coerce')
                except:
                    pass
                df[col] = df[col].applymap(lambda x: np.nan if isinstance(x, str) else x)
                df = df.replace(r'^\s*$', np.nan, regex=True)
                df = df.replace(r'^([A-Za-z])+$', np.nan, regex=True)
                # df = df.replace(':', np.nan, regex=True)
                df = df.dropna(how='all')
        
        if 'Date' in df.columns:
            df['Datetime'] = np.nan
            try:
                df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.time
            except ValueError:
                for i, row in df.iterrows():
                    # print(i, row)
                    try:
                        df['Time'].loc[i] = (dt.datetime(2010,1,1) + pd.Timedelta(df['Time'].loc[i]*1440, unit='m')).time()
                    except:
                        pass
            except:
                df['Time'] = pd.to_datetime(dt.datetime(2010,1,1) + pd.TimedeltaIndex(df['Time']*1440, unit='m')).time
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except:
                for i, row in df.iterrows():
                    # print(row)
                    df['Date'][i] = dp.parse(df['Date'][i], languages=['es'])
            for i, row in df.iterrows():
                df['Datetime'][i] = dt.datetime.combine(df['Date'][i], df['Time'][i])
            df.index = df['Datetime']
            df = df.drop(['Date', 'Time', 'Datetime'], axis=1)
            df.index.name = None
        else:
            if df['Datetime'].dtype != '<M8[ns]':
                try:
                    # df['Datetime'] = df['Datetime'].str.replace(' a.','')
                    # df['Datetime'] = df['Datetime'].str.replace(' p.','')
                    # df['Datetime'] = df['Datetime'].str.replace('.',':')
                    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y/%m/%d %H:%M:%S')
                except:
                    pass
                try:
                    for i, row in df.iterrows():
                        # print(row)
                        df['Datetime'][i] = dp.parse(df['Datetime'][i], languages=['es'])
                    df.index = df['Datetime']
                    df = df.drop(['Datetime'], axis=1)
                    df.index.name = None
                except:
                    df.index = df['Datetime']
                    df = df.drop(['Datetime'], axis=1)
                    df.index.name = None
            else:
                df.index = df['Datetime']
                df = df.drop(['Datetime'], axis=1)
                df.index.name = None
                # print('Datetime Format OK')
    
        df = df.sort_index()
        df = df.dropna(how='all')
        df.to_csv('../CSV/'+fname_csv_final)

    except:
        print('ERROR', ema, year)
        pass


#Add files with datetime errors individual same days
f = open('../DATETIME_SAME_ERRORS_IND.txt', 'r')
# f = open('../DATETIME_SAME_ERRORS_IND_FINAL.txt', 'r')
errors = f.readlines()
errors = list(map(lambda x: x.replace('\n', ''), errors))
errors = list(map(lambda x: x.replace('DATETIME ERROR ', ''), errors))
errors = list(map(lambda x: x.replace('ERROR ', ''), errors))

dif = pd.read_csv('../diff_stations.csv', encoding= 'unicode_escape')
dif = dif.drop(['Unnamed: 0', 'id'], axis=1)
difm = pd.read_csv('../diff_stations_mod.csv', encoding= 'unicode_escape')
difm = difm.drop(['i', 'id'], axis=1)

# f = open('../DATETIME_SAME_ERRORS_IND_FINAL.txt', 'w')

for error in errors:
    print(error)
    # error = 'AGUASCALIENTES_2006.csv 5484'
    # error = errors[0]
    # ema = error[:-4]
    ema = error.split('.csv', 1)[0]
    if ema[-4:].isnumeric():
        year = ema[-4:]
        ema = ema[:-5]
        file = [file for file in os.listdir('./'+year) if file.startswith(ema+'_'+year)]
    elif ema[-2:].isnumeric():
        month = ema[-2:]
        year = ema[-7:-3]
        ema = ema[:-8]
        file = [file for file in os.listdir('./'+year) if file.startswith(ema+'_'+year+'_'+month)]
    else:
        pass
    
    file = file[0]
    
    try:
        try:
            fname_xls = file
            workbook = xlrd.open_workbook('./'+year+'/'+fname_xls, logfile=open(os.devnull, "w"))
            df = pd.read_excel(workbook)
            fname_csv_final = fname_xls[:-4]+'.csv'
        except XLRDError:
            try:
                fname_xls = file
                df = pd.read_excel('./'+year+'/'+fname_xls)
                fname_csv_final = fname_xls[:-4]+'.csv'
            except ValueError:
                fname_csv = file
                df = pd.read_csv('./'+year+'/'+fname_csv, encoding= 'unicode_escape')
                fname_csv_final = file
        except:
            fname_xlsx = file+'x'
            workbook = xlrd.open_workbook('./'+year+'/'+fname_xlsx, logfile=open(os.devnull, "w"))
            df = pd.read_excel(workbook)
            fname_xlsx[:-5]+'.csv'
        
        a = np.empty(26-len(df.columns.values))
        a[:] = np.nan
        cols = pd.DataFrame(np.concatenate([df.columns.values,a])).T
        
        for i, row in dif.iterrows():
            # i = 22
            # row = dif.loc[i]
            x = dif.loc[i].dropna().values
            y = cols.loc[0].dropna().values.astype(str)
            check = np.array_equal(x,y)
            # print(i, check)
            if check == True:
                df.columns = difm.loc[i].dropna().values
                col = list(df.columns.values)
                for x in ['Date', 'Time', 'Datetime']:
                    if x in col:
                        col.remove(x)
                try:
                    for c in col:
                        df[c] = pd.to_numeric(df[c], errors='coerce')
                except:
                    pass
                df[col] = df[col].applymap(lambda x: np.nan if isinstance(x, str) else x)
                df = df.replace(r'^\s*$', np.nan, regex=True)
                df = df.replace(r'^([A-Za-z])+$', np.nan, regex=True)
                # df = df.replace(':', np.nan, regex=True)
                df = df.dropna(how='all')
        
        if 'Date' in df.columns:
            df['Datetime'] = np.nan
            try:
                df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.time
            except ValueError:
                for i, row in df.iterrows():
                    # print(i, row)
                    try:
                        df['Time'].loc[i] = (dt.datetime(2010,1,1) + pd.Timedelta(df['Time'].loc[i]*1440, unit='m')).time()
                    except:
                        pass
            except:
                df['Time'] = pd.to_datetime(dt.datetime(2010,1,1) + pd.TimedeltaIndex(df['Time']*1440, unit='m')).time
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except:
                for i, row in df.iterrows():
                    # print(row)
                    df['Date'][i] = dp.parse(df['Date'][i], languages=['es'])
            for i, row in df.iterrows():
                df['Datetime'][i] = dt.datetime.combine(df['Date'][i], df['Time'][i])
            df.index = df['Datetime']
            df = df.drop(['Date', 'Time', 'Datetime'], axis=1)
            df.index.name = None
        else:
            if df['Datetime'].dtype != '<M8[ns]':
                df['Datetime'] = df['Datetime'].str.replace('a.m.','AM')
                df['Datetime'] = df['Datetime'].str.replace('p.m.','PM')
                df['Datetime'] = df['Datetime'].str.replace('.',':')
                df['Datetime'] = df['Datetime'].str.replace('a:','AM')
                df['Datetime'] = df['Datetime'].str.replace('p:','PM')
                try:                    
                    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y/%m/%d %I:%M:%S %p')
                except:
                    try:
                        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %I:%M:%S %p')
                    except:
                        try:
                            df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y/%m/%d %H:%M')
                        except:
                            try:
                                df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H:%M')
                            except:
                                try:
                                    df['Datetime'] = pd.to_datetime(df['Datetime'])
                                except:
                                    pass
                try:
                    for i, row in df.iterrows():
                        # print(row)
                        df['Datetime'][i] = dp.parse(df['Datetime'][i], languages=['es'])
                    df.index = df['Datetime']
                    df = df.drop(['Datetime'], axis=1)
                    df.index.name = None
                except:
                    df.index = df['Datetime']
                    df = df.drop(['Datetime'], axis=1)
                    df.index.name = None
            else:
                df.index = df['Datetime']
                df = df.drop(['Datetime'], axis=1)
                df.index.name = None
                # print('Datetime Format OK')
    
        df = df.sort_index()
        df = df.dropna(how='all')
        df.to_csv('../CSV/'+fname_csv_final)
        
    except:
        # print('ERROR', error, file=f)
        print('ERROR', error)
        pass
f.close()




#Merge all files in one for each station for emas with datetime errors
f = open('../DATETIME_ERRORS.txt', 'r')
errors = f.readlines()
errors = list(map(lambda x: x.replace('\n', ''), errors))
errors = list(map(lambda x: x.replace('DATETIME ERROR ', ''), errors))

dif = pd.read_csv('../diff_stations.csv', encoding= 'unicode_escape')
dif = dif.drop(['Unnamed: 0', 'id'], axis=1)
difm = pd.read_csv('../diff_stations_mod.csv', encoding= 'unicode_escape')
difm = difm.drop(['i', 'id'], axis=1)

emas = ['']
emas_files = os.listdir('../CSV/')
emas_files.remove('COMPLETAS')
emas_files.remove('DIFERENTES')
ema_complete = pd.DataFrame()

# errors = ['ZIMAPAN.csv']

for error in errors:
    print(error)
    # error = errors[0]
    # ema = error[:-4]
    ema_name = error[:-4]
    emas_files = [file for file in os.listdir('../CSV/') if file.startswith(ema_name)]

    for ema_file in emas_files:
        # print(ema_file)
        # ema = 'CANCUN_2013_09.csv'
        # ema_file = emas_files[0]
        df = pd.read_csv('../CSV/'+ema_file, parse_dates=['Unnamed: 0'])
        df.index = df['Unnamed: 0']
        df = df.drop(['Unnamed: 0'], axis=1)
        df.index.name = None
        df = df.dropna(how='all')
        
        ema = ema_file[:-4]
        if ema[-4:].isnumeric():
            ema = ema[:-5]
        elif ema[-2:].isnumeric():
            ema = ema[:-8]
        else:
            pass
        try:
            ema = ema.replace('-',' ')
            # ema = ema.replace('_','')
        except:
            pass
        # print(ema)
        if ema == emas[-1]:
            # print('append file')
            ema_complete = pd.concat([ema_complete, df])
            emas.append(ema)
        else:
            if ema not in emas:
                print('Write file '+emas[-1])
                ema_complete.to_csv('../CSV/COMPLETAS/'+emas[-1]+'.csv')
                ema_complete = pd.DataFrame()
                
                print(ema)
                ema_complete = pd.concat([ema_complete, df])
                emas.append(ema)
os.remove('../CSV/COMPLETAS/.csv')
    

#Merge all files in one for each station for emas with datetime errors SAME
f = open('../DATETIME_SAME_ERRORS.txt', 'r')
errors = f.readlines()
errors = list(map(lambda x: x.replace('\n', ''), errors))
errors = list(map(lambda x: x.replace('DATETIME ERROR ', ''), errors))

dif = pd.read_csv('../diff_stations.csv', encoding= 'unicode_escape')
dif = dif.drop(['Unnamed: 0', 'id'], axis=1)
difm = pd.read_csv('../diff_stations_mod.csv', encoding= 'unicode_escape')
difm = difm.drop(['i', 'id'], axis=1)

emas = ['']
emas_files = os.listdir('../CSV/')
emas_files.remove('COMPLETAS')
emas_files.remove('DIFERENTES')
emas_files.remove('DATETIME')
ema_complete = pd.DataFrame()

# errors = ['ZIMAPAN.csv']

for error in errors:
    print(error)
    # error = errors[0]
    # ema = error[:-4]
    # ['ESCUINTLA', 'IXTLAN DEL RIO', 'NUEVA ROSITA', 'PARAISO', 'URUAPAN', 'ZIHUATANEJO']
    # ema_name = 'ZIHUATANEJO'
    ema_name = error.split('.csv', 1)[0]
    emas_files = [file for file in os.listdir('../CSV/') if file.startswith(ema_name)]

    for ema_file in emas_files:
        # print(ema_file)
        # ema = 'CANCUN_2013_09.csv'
        # ema_file = emas_files[0]
        df = pd.read_csv('../CSV/'+ema_file, parse_dates=['Unnamed: 0'])
        df.index = df['Unnamed: 0']
        df = df.drop(['Unnamed: 0'], axis=1)
        df.index.name = None
        df = df.dropna(how='all')
        df = df[~df.index.duplicated(keep='last')]
        
        ema = ema_file[:-4]
        if ema[-4:].isnumeric():
            ema = ema[:-5]
        elif ema[-2:].isnumeric():
            ema = ema[:-8]
        else:
            pass
        try:
            ema = ema.replace('-',' ')
            # ema = ema.replace('_','')
        except:
            pass
        # print(ema)
        if ema == emas[-1]:
            # print('append file')
            ema_complete = pd.concat([ema_complete, df])
            emas.append(ema)
        else:
            if ema not in emas:
                print('Write file '+emas[-1])
                ema_complete = ema_complete[~ema_complete.index.duplicated(keep='last')]
                ema_complete.to_csv('../CSV/COMPLETAS/'+emas[-1]+'.csv')
                ema_complete = pd.DataFrame()
                
                print(ema)
                ema_complete = pd.concat([ema_complete, df])
                emas.append(ema)
                
ema_complete = ema_complete[~ema_complete.index.duplicated(keep='last')]
ema_complete.to_csv('../CSV/COMPLETAS/'+emas[-1]+'.csv')
os.remove('../CSV/COMPLETAS/.csv')
    



























#Add files with datetime errors individual same days
f = open('../DATETIME_SAME_ERRORS_IND.txt', 'r')
# f = open('../DATETIME_SAME_ERRORS_IND_FINAL.txt', 'r')
errors = f.readlines()
errors = list(map(lambda x: x.replace('\n', ''), errors))
errors = list(map(lambda x: x.replace('DATETIME ERROR ', ''), errors))

dif = pd.read_csv('../diff_stations.csv', encoding= 'unicode_escape')
dif = dif.drop(['Unnamed: 0', 'id'], axis=1)
difm = pd.read_csv('../diff_stations_mod.csv', encoding= 'unicode_escape')
difm = difm.drop(['i', 'id'], axis=1)

f = open('../DATETIME_SAME_ERRORS_IND_FINAL.txt', 'w')

for error in errors:
    print(error)
    # error = 'CIUDAD MANTE_2005_09.csv 4307'
    # error = errors[0]
    # ema = error[:-4]
    ema = error.split('.csv', 1)[0]
    if ema[-4:].isnumeric():
        year = ema[-4:]
        ema = ema[:-5]
        file = [file for file in os.listdir('./'+year) if file.startswith(ema+'_'+year)]
    elif ema[-2:].isnumeric():
        month = ema[-2:]
        year = ema[-7:-3]
        ema = ema[:-8]
        file = [file for file in os.listdir('./'+year) if file.startswith(ema+'_'+year+'_'+month)]
    else:
        pass
    
    file = file[0]
    
    try:
        try:
            fname_xls = file
            workbook = xlrd.open_workbook('./'+year+'/'+fname_xls, logfile=open(os.devnull, "w"))
            df = pd.read_excel(workbook)
            fname_csv_final = fname_xls[:-4]+'.csv'
        except XLRDError:
            try:
                fname_xls = file
                df = pd.read_excel('./'+year+'/'+fname_xls)
                fname_csv_final = fname_xls[:-4]+'.csv'
            except ValueError:
                fname_csv = file
                df = pd.read_csv('./'+year+'/'+fname_csv, encoding= 'unicode_escape')
                fname_csv_final = file
        except:
            fname_xlsx = file+'x'
            workbook = xlrd.open_workbook('./'+year+'/'+fname_xlsx, logfile=open(os.devnull, "w"))
            df = pd.read_excel(workbook)
            fname_xlsx[:-5]+'.csv'
        
        a = np.empty(26-len(df.columns.values))
        a[:] = np.nan
        cols = pd.DataFrame(np.concatenate([df.columns.values,a])).T
        
        for i, row in dif.iterrows():
            # i = 22
            # row = dif.loc[i]
            x = dif.loc[i].dropna().values
            y = cols.loc[0].dropna().values.astype(str)
            check = np.array_equal(x,y)
            # print(i, check)
            if check == True:
                df.columns = difm.loc[i].dropna().values
                col = list(df.columns.values)
                for x in ['Date', 'Time', 'Datetime']:
                    if x in col:
                        col.remove(x)
                try:
                    for c in col:
                        df[c] = pd.to_numeric(df[c], errors='coerce')
                except:
                    pass
                df[col] = df[col].applymap(lambda x: np.nan if isinstance(x, str) else x)
                df = df.replace(r'^\s*$', np.nan, regex=True)
                df = df.replace(r'^([A-Za-z])+$', np.nan, regex=True)
                # df = df.replace(':', np.nan, regex=True)
                df = df.dropna(how='all')
        
        if 'Date' in df.columns:
            df['Datetime'] = np.nan
            try:
                df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.time
            except ValueError:
                for i, row in df.iterrows():
                    # print(i, row)
                    try:
                        df['Time'].loc[i] = (dt.datetime(2010,1,1) + pd.Timedelta(df['Time'].loc[i]*1440, unit='m')).time()
                    except:
                        pass
            except:
                df['Time'] = pd.to_datetime(dt.datetime(2010,1,1) + pd.TimedeltaIndex(df['Time']*1440, unit='m')).time
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except:
                for i, row in df.iterrows():
                    # print(row)
                    df['Date'][i] = dp.parse(df['Date'][i], languages=['es'])
            for i, row in df.iterrows():
                df['Datetime'][i] = dt.datetime.combine(df['Date'][i], df['Time'][i])
            df.index = df['Datetime']
            df = df.drop(['Date', 'Time', 'Datetime'], axis=1)
            df.index.name = None
        else:
            if df['Datetime'].dtype != '<M8[ns]':
                df['Datetime'] = df['Datetime'].str.replace('a.m.','AM')
                df['Datetime'] = df['Datetime'].str.replace('p.m.','PM')
                df['Datetime'] = df['Datetime'].str.replace('.',':')
                df['Datetime'] = df['Datetime'].str.replace('a:','AM')
                df['Datetime'] = df['Datetime'].str.replace('p:','PM')
                try:                    
                    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y/%m/%d %I:%M:%S %p')
                except:
                    try:
                        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %I:%M:%S %p')
                    except:
                        try:
                            df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y/%m/%d %H:%M')
                        except:
                            try:
                                df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H:%M')
                            except:
                                try:
                                    df['Datetime'] = pd.to_datetime(df['Datetime'])
                                except:
                                    pass
                try:
                    for i, row in df.iterrows():
                        # print(row)
                        df['Datetime'][i] = dp.parse(df['Datetime'][i], languages=['es'])
                    df.index = df['Datetime']
                    df = df.drop(['Datetime'], axis=1)
                    df.index.name = None
                except:
                    df.index = df['Datetime']
                    df = df.drop(['Datetime'], axis=1)
                    df.index.name = None
            else:
                df.index = df['Datetime']
                df = df.drop(['Datetime'], axis=1)
                df.index.name = None
                # print('Datetime Format OK')
    
        df = df.sort_index()
        df = df.dropna(how='all')
        df = df[~df.index.duplicated(keep='last')]
        df.to_csv('../CSV/DATETIME/'+fname_csv_final)
        
    except:
        print('ERROR', error, file=f)
        # print('ERROR', error)
        pass
f.close()










dif = pd.read_csv('../diff_stations.csv', encoding= 'unicode_escape')
dif = dif.drop(['Unnamed: 0', 'id'], axis=1)
difm = pd.read_csv('../diff_stations_mod.csv', encoding= 'unicode_escape')
difm = difm.drop(['i', 'id'], axis=1)

#Arrange EMAS data in CSV file with same format
# f = open('../ERRORS.txt', 'w')
for year in years:
    # year = '2020'
    emas_in_year = os.listdir('./'+year)
    for ema in emas_in_year:
        # i += 1
        print(ema)
        # ema = 'QUERETARO CONAGUA_2020_01.csv'
        # ema = 'CHETUMAL_2011.csv'
        # ema = emas_in_year[0]
        try:
            try:
                fname_xls = ema
                workbook = xlrd.open_workbook('./'+year+'/'+fname_xls, logfile=open(os.devnull, "w"))
                df = pd.read_excel(workbook)
                fname_csv_final = fname_xls[:-4]+'.csv'
            except XLRDError:
                try:
                    fname_xls = ema
                    df = pd.read_excel('./'+year+'/'+fname_xls)
                    fname_csv_final = fname_xls[:-4]+'.csv'
                except ValueError:
                    fname_csv = ema
                    df = pd.read_csv('./'+year+'/'+fname_csv, encoding= 'unicode_escape')
                    fname_csv_final = ema
            except:
                fname_xlsx = ema+'x'
                workbook = xlrd.open_workbook('./'+year+'/'+fname_xlsx, logfile=open(os.devnull, "w"))
                df = pd.read_excel(workbook)
                fname_xlsx[:-5]+'.csv'
            
            a = np.empty(26-len(df.columns.values))
            a[:] = np.nan
            cols = pd.DataFrame(np.concatenate([df.columns.values,a])).T
            
            for i, row in dif.iterrows():
                # i = 22
                # row = dif.loc[i]
                x = dif.loc[i].dropna().values
                y = cols.loc[0].dropna().values.astype(str)
                check = np.array_equal(x,y)
                # print(i, check)
                if check == True:
                    df.columns = difm.loc[i].dropna().values
                    col = list(df.columns.values)
                    for x in ['Date', 'Time', 'Datetime']:
                        if x in col:
                            col.remove(x)
                    try:
                        for c in col:
                            df[c] = pd.to_numeric(df[c], errors='coerce')
                    except:
                        pass
                    df[col] = df[col].applymap(lambda x: np.nan if isinstance(x, str) else x)
                    df = df.replace(r'^\s*$', np.nan, regex=True)
                    df = df.replace(r'^([A-Za-z])+$', np.nan, regex=True)
                    df = df.dropna(how='all')
                    # df = df[1:]
                    # df.columns
            
            if 'Date' in df.columns:
                df['Datetime'] = np.nan
                try:
                    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.time
                except ValueError:
                    for i, row in df.iterrows():
                        # print(i, row)
                        try:
                            df['Time'].loc[i] = (dt.datetime(2010,1,1) + pd.Timedelta(df['Time'].loc[i]*1440, unit='m')).time()
                        except:
                            pass
                except:
                    df['Time'] = pd.to_datetime(dt.datetime(2010,1,1) + pd.TimedeltaIndex(df['Time']*1440, unit='m')).time
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                except:
                    for i, row in df.iterrows():
                        # print(row)
                        df['Date'][i] = dp.parse(df['Date'][i], languages=['es'])
                for i, row in df.iterrows():
                    df['Datetime'][i] = dt.datetime.combine(df['Date'][i], df['Time'][i])
                df.index = df['Datetime']
                df = df.drop(['Date', 'Time', 'Datetime'], axis=1)
                df.index.name = None
            else:
                if df['Datetime'].dtype != '<M8[ns]':
                    df['Datetime'] = df['Datetime'].str.replace('a.m.','AM')
                    df['Datetime'] = df['Datetime'].str.replace('p.m.','PM')
                    df['Datetime'] = df['Datetime'].str.replace('.',':')
                    df['Datetime'] = df['Datetime'].str.replace('a:','AM')
                    df['Datetime'] = df['Datetime'].str.replace('p:','PM')
                    try:                    
                        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y/%m/%d %I:%M:%S %p')
                    except:
                        try:
                            df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %I:%M:%S %p')
                        except:
                            try:
                                df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y/%m/%d %H:%M')
                            except:
                                try:
                                    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H:%M')
                                except:
                                    try:
                                        df['Datetime'] = pd.to_datetime(df['Datetime'])
                                    except:
                                        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y%m%d %H%M')
                                        pass
                    try:
                        for i, row in df.iterrows():
                            # print(row)
                            df['Datetime'][i] = dp.parse(df['Datetime'][i], languages=['es'])
                        df.index = df['Datetime']
                        df = df.drop(['Datetime'], axis=1)
                        df.index.name = None
                    except:
                        df.index = df['Datetime']
                        df = df.drop(['Datetime'], axis=1)
                        df.index.name = None
                else:
                    df.index = df['Datetime']
                    df = df.drop(['Datetime'], axis=1)
                    df.index.name = None
                    # print('Datetime Format OK')
        
            df = df.sort_index()
            df = df.dropna(how='all')
            df.to_csv('../CSV/'+fname_csv_final)
        except:
            pass



#Merge all files in one for each station for emas with datetime errors SAME

dif = pd.read_csv('../diff_stations.csv', encoding= 'unicode_escape')
dif = dif.drop(['Unnamed: 0', 'id'], axis=1)
difm = pd.read_csv('../diff_stations_mod.csv', encoding= 'unicode_escape')
difm = difm.drop(['i', 'id'], axis=1)

emas = ['']
emas_files = os.listdir('../CSV/')
emas_files.remove('COMPLETAS')
emas_files.remove('DIFERENTES')
emas_files.remove('DATETIME')
ema_complete = pd.DataFrame()

# errors = ['ZIMAPAN.csv']

for error in errors:
    print(error)
    # error = errors[0]
    # ema = error[:-4]
    # ['ESCUINTLA', 'IXTLAN DEL RIO', 'NUEVA ROSITA', 'PARAISO', 'URUAPAN', 'ZIHUATANEJO']
    # ema_name = 'ACAPULCO'
    # ema_name = error.split('.csv', 1)[0]
    emas_files = [file for file in os.listdir('../CSV/') if file.startswith(ema_name)]

    for ema_file in emas_files:
        # print(ema_file)
        # ema = 'CANCUN_2013_09.csv'
        # ema_file = emas_files[0]
        df = pd.read_csv('../CSV/'+ema_file, parse_dates=['Unnamed: 0'])
        df.index = df['Unnamed: 0']
        df = df.drop(['Unnamed: 0'], axis=1)
        df.index.name = None
        df = df.dropna(how='all')
        df = df[~df.index.duplicated(keep='last')]
        
        ema = ema_file[:-4]
        if ema[-4:].isnumeric():
            ema = ema[:-5]
        elif ema[-2:].isnumeric():
            ema = ema[:-8]
        else:
            pass
        try:
            ema = ema.replace('-',' ')
            # ema = ema.replace('_','')
        except:
            pass
        # print(ema)
        if ema == emas[-1]:
            # print('append file')
            ema_complete = pd.concat([ema_complete, df])
            emas.append(ema)
        else:
            if ema not in emas:
                print('Write file '+emas[-1])
                ema_complete = ema_complete[~ema_complete.index.duplicated(keep='last')]
                ema_complete.to_csv('../CSV/COMPLETAS/'+emas[-1]+'.csv')
                ema_complete = pd.DataFrame()
                
                print(ema)
                ema_complete = pd.concat([ema_complete, df])
                emas.append(ema)
                
ema_complete = ema_complete[~ema_complete.index.duplicated(keep='last')]
ema_complete.to_csv('../CSV/COMPLETAS/'+emas[-1]+'.csv')
os.remove('../CSV/COMPLETAS/.csv')
    















# f = open('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/EMAS_ERRORS.txt', 'w')

emas_names = pd.read_csv('D:/DANI/VARIOS/EMAS/names_emas.csv', encoding='latin1')
emas_names = emas_names.dropna(subset=['names2'])
emas_names.index = emas_names['names2']
emas_names.index.name = None
emas_names = emas_names.drop(['names2', 'names3'], axis=1)

emas1 = os.listdir('D:/DANI/VARIOS/EMAS/COMPLETAS/')
emas2 = os.listdir('D:/DANI/VARIOS/EMAS/DATOS_ORIGINALES/CSV/COMPLETAS/')

for ema2, row in emas_names.iterrows():
    print(row[0], ema2)
    # ema2 = 'ACAPONETA'
    # ema1 = 'ACAPONETA'
    ema1 = row[0]
    
    try:
        dfema1 = pd.read_csv('D:/DANI/VARIOS/EMAS/COMPLETAS/'+ema1+'_All.csv', encoding='latin1')
        dfema1.index = dfema1['Unnamed: 0']
        dfema1.index.name = None
        dfema1 = dfema1.drop(['Unnamed: 0'], axis=1)
        dfema1['P'][dfema1['P']>250] = np.nan
        
        dfema2 = pd.read_csv('D:/DANI/VARIOS/EMAS/DATOS_ORIGINALES/CSV/COMPLETAS/'+ema2+'.csv', encoding='latin1')
        dfema2.index = dfema2['Unnamed: 0']
        dfema2.index.name = None
        dfema2 = dfema2.drop(['Unnamed: 0'], axis=1)
        dfema2['P'][dfema2['P']>250] = np.nan
        
        df = pd.concat([dfema1, dfema2])
        df = df.sort_index()
        df1 = df[~df['P'].isnull()]
        df1 = df1[~df1.index.duplicated(keep='last')]
        df = pd.concat([df1, df], axis=0)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        
        df.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/EMAS/'+ema2+'_All.csv')
        
    except:
        print(row[0], ema2) #, file=f)
        dfema2 = pd.read_csv('D:/DANI/VARIOS/EMAS/DATOS_ORIGINALES/CSV/COMPLETAS/'+ema2+'.csv', encoding='latin1')
        dfema2.index = dfema2['Unnamed: 0']
        dfema2.index.name = None
        dfema2 = dfema2.drop(['Unnamed: 0'], axis=1)
        dfema2['P'][dfema2['P']>250] = np.nan
        
        df = dfema2.sort_index()
        df1 = df[~df['P'].isnull()]
        df1 = df1[~df1.index.duplicated(keep='last')]
        df = pd.concat([df1, df], axis=0)
        df = df[~df.index.duplicated(keep='first')]
        # df = df[~df.index.duplicated(keep='last')]
        
        df.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/EMAS/'+ema2+'_All.csv')
# f.close()


emas_names = pd.read_csv('D:/DANI/VARIOS/EMAS/names_emas.csv', encoding='latin1')
emas_names = emas_names.drop(['names3'], axis=1)
emas_names = emas_names[emas_names.isnull().any(1)]
emas_names = emas_names.drop(['names2'], axis=1)

emas1 = os.listdir('D:/DANI/VARIOS/EMAS/COMPLETAS/')

for ema1, row in emas_names.iterrows():
    print(row[0])
    ema = row[0]
    
    dfema1 = pd.read_csv('D:/DANI/VARIOS/EMAS/COMPLETAS/'+ema+'_All.csv', encoding='latin1')
    dfema1.index = dfema1['Unnamed: 0']
    dfema1.index.name = None
    dfema1 = dfema1.drop(['Unnamed: 0'], axis=1)
    dfema1['P'][dfema1['P']>250] = np.nan
    
    df = dfema1.sort_index()
    df1 = df[~df['P'].isnull()]
    df1 = df1[~df1.index.duplicated(keep='last')]
    df = pd.concat([df1, df], axis=0)
    df = df[~df.index.duplicated(keep='first')]
    
    if len(df) != 0:
        df.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/EMAS/'+ema+'_All.csv')
        
        
        
        
        
f = open('D:/DANI/VARIOS/EMAS/DATOS_ORIGINALES/CSV/DT_ERRORS.txt', 'w')

emas_ind = os.listdir('D:/DANI/VARIOS/EMAS/DATOS_ORIGINALES/CSV/')
rem = ['COMPLETAS', 'DATETIME', 'DIFERENTES']
for x in rem:
    emas_ind.remove(x)

for emas in emas_ind:
    # emas = emas_ind[0]
    ema = pd.read_csv('D:/DANI/VARIOS/EMAS/DATOS_ORIGINALES/CSV/'+emas)
    
    try:
        mean_len = np.mean([len(i) for i in ema['Unnamed: 0']])    
        if mean_len>19:
            print(emas, file=f)
    except:
        print(emas, file=f)
        pass

f.close()
        
        
        
        
#FILES WITH DATETIME ERRORS
emas_dt_errors = ['ACAPULCO', 'CAMPECHE']
check = ['ANGAMACUTIRO', 'CEMCAS', 'CHINIPAS', 'CHOIX', 'CIUDAD CONSTITUCION', 'COLIMA', 'CULIACAN', 'KANTUNILKIN', 'LA PAZ', 'LOS TUXTLAS II', 'MANZANILLO',
         'MARISMAS NACIONALES', 'MORELIA', 'NEVADO DE COLIMA', 'NUEVO CASAS GRANDES', 'PANTANOS DE CENTLA', 'PETACALCO', 'PIEDRAS NEGRAS', 'PINOTEPA', 'PRESA LA CANGREJERA',
         'PROGRESO', 'PUERTO ESCONDIDO', 'RIO LAGARTOS', 'RIO VERDE', 'SIAN KAAN II', 'TACUBAYA', 'TAMPICO MADERO', 'TAMPICO', 'TEMOSACHIC', ]
wrong_P_values= ['BENITO JUAREZ', 'CABO PULMO']

combine = 'COATZACOALCO', 'COATZACOALCOS' '?'


revisar = ['CHAPALA en adelante']


