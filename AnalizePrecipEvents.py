# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:08:17 2022

@author: HIDRAULICA-Dani
"""

import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
from matplotlib.collections import LineCollection

os.chdir('D:/DANI/2023/TEMA_TORMENTAS/DATOS')
emas_files = os.listdir('./EMAS')
emas_events = os.listdir('./EMAS_Events_6,20') #EMAS_Events

# emas_files = []
# for file in files:
#     if file.endswith(".csv"):
#         emas_files.append(file)

# df_ind = pd.DataFrame()
# df_prom = pd.DataFrame()
# df_max = pd.DataFrame()
# df_min = pd.DataFrame()
# i = 0

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings
    
    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection
    
    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)
    
    Returns
    -------
    lc : LineCollection instance.
    """
    
    # find axes
    ax = plt.gca() if ax is None else ax
    
    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)
    
    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))
    
    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


dur_mean = []
dfdurs = pd.DataFrame([0,0,0,0,0], [0.5,2.0,3.5,5.0,6.5], columns=['x'])
dfquartile = pd.DataFrame([0,0,0,0], [1,2,3,4], columns=['x'])

for ema_event in emas_events:
# for ema_file in emas_files:
    # i += 1
    # ema_file = 'MONTERREY_All.csv'
    # ema_file = emas_files[0]
    # ema_name = ema_file[:-8]
    # folder_name = ema_name+'_Events'
    # ema_event = emas_events[1]
    ema_name = ema_event[:-7]
    folder_name = ema_event
    ema_file = ema_name+'_All.csv'
    
    events = os.listdir('./EMAS_Events/'+folder_name)

    ema = pd.read_csv('./EMAS/'+ema_file, parse_dates=['Unnamed: 0'])
    ema.index = ema['Unnamed: 0']
    ema = ema.drop(['Unnamed: 0'], axis=1)
    ema.index.name = None

    j = 0
    newindex = np.linspace(0.0, 1.0, 10001)
    newpindex = np.linspace(0.0, 10000.0, 1001)
    dfpall = pd.DataFrame()
    dfall = pd.DataFrame()
    tp_list = []
    d_list = []
    sdt_list = []
    dt_tpm10_list = []
    tpm10_cum_list = []
    tpm10_list = []
    im10_list = []
    im60_list = []
    loc_dt_quart = []
    
    # if len(events)<
    
    for event in events:
        j += 1
        # event = events[277]
        # event = 'ACAPULCO_Event_277.csv'
        df1 = pd.read_csv('./EMAS_Events/'+folder_name+'/'+event, parse_dates=['t'])
        df1.index = df1['t']
        df1 = df1.drop(['t'], axis=1)
        df1.index.name = None
        # print(event, len(df1))
        
        if df1.max()[0] > 250:
            print(event, 'deleted')
            os.remove('./EMAS_Events/'+folder_name+'/'+event)
            continue
        
        if len(df1) > 0:
        # if len(df1) <= 3:
        # if len(df1) <= 6:
        # if len(df1) <= 12:
        # if ((len(df1) > 6) & (len(df1) <= 12)):
        # if ((len(df1) > 12) & (len(df1) <= 18)):
        # if ((len(df1) > 18) & (len(df1) <= 24)):
        # if ((len(df1) > 12) & (len(df1) <= 24)):
        # if len(df1) > 6:
        # if len(df1) > 12:
        # if len(df1) > 24:
        
            start_dt = df1.index[0]
            
            if len(df1.index) == 1:
                end_dt = df1.index[0] + datetime.timedelta(minutes=60)
            else:
                end_dt = df1.index[-1] + datetime.timedelta(minutes=60)
            
            df = ema['P'][start_dt:end_dt].to_frame()
            dfcum = df.cumsum()
            
            # dfcum.iloc[0].values[0]
            
            for row, p in dfcum.iterrows():
                # print(row, p)
                if p['P'] == 0 or np.isnan(p['P']):
                    df = df.drop(row)
                    dfcum = dfcum.drop(row)
                    # print(row, p)
            
            dfcum_rev = dfcum.iloc[::-1]
            dfcum_revs = dfcum_rev.shift(-1)
            
            for row, p in dfcum_rev.iterrows():
                # print(row, p)
                if p['P'] == dfcum_revs.loc[row]['P']:
                    df = df.drop(row)
                    dfcum = dfcum.drop(row)
                    # print(row, p)
            
            try:
                delta = df.index[1] - df.index[0]
            except:
                continue
        
            df_0 = pd.DataFrame([0], index=[df.index[0]-delta], columns=['P'])
            # df_1 = pd.DataFrame([0], index=[df.index[-1]+delta], columns=['P'])
            df = pd.concat([df_0, df]) #, df_1])
            
            dfcum_0 = pd.DataFrame([0], index=[dfcum.index[0]-delta], columns=['P'])
            # dfcum_1 = pd.DataFrame(dfcum.iloc[-1]['P'], index=[dfcum.index[-1]+delta], columns=['P'])
            dfcum = pd.concat([dfcum_0, dfcum]) #, dfcum_1])
            
            tp = df.sum().values[0]
            d = df.index[-1] - df.index[0]
            d = d.days*24 + d.seconds/3600
            sdt = df.index[0]
            dt_tpm10 = df[df==df.max()].dropna().index[0]
            tpm10_cum = dfcum.loc[dt_tpm10][0]
            tpm10 = df.max().values[0]
            im10 = tpm10*6
            im60 = df.rolling(6).sum().max().values[0]
            
            tp_list.append(tp)
            d_list.append(d)
            sdt_list.append(sdt)
            dt_tpm10_list.append(dt_tpm10)
            tpm10_cum_list.append(tpm10_cum)
            tpm10_list.append(tpm10)
            im10_list.append(im10)
            im60_list.append(im60)
            
            P = df['P'].values
            Ps = dfcum['P'].values
            # Pn = P/P.max()
            Psn = np.round(Ps/Ps[-1], 5)
            
            # delta = (df.index[-1].timestamp() - df.index[0].timestamp())/60
            # t = np.arange(0.0, len(Psn)*10, 10.0)
            t = []
            for datet in df.index:
                dur = datet - df.index[0]
                dur = dur.days*24 + dur.seconds/3600
                t.append(dur)
            t = np.asarray(t)
            tn = np.round(t/t.max(), 5)
            
            dfp = pd.DataFrame(Ps, t)
            dfps = pd.DataFrame(Psn, tn)
            
            # plt.plot(dfp)
            # plt.plot(dfps)
            # print(event, len(df1), start_dt, df['P'].max(), df['P'].resample('1H').sum().max(), df['P'].sum())
            
            #Interpolate to new index
            dfp_reindexed = dfp.reindex(index = newpindex)
            dfp_reindexed.interpolate(method = 'linear', inplace = True)
            dfp_reindexed.columns = [str(j)] #(str(i)+str(j))
            dfpall = pd.concat([dfpall, dfp_reindexed], axis=1)
            
            # plt.plot(dfall)
            
            #Interpolate to new index
            dfps_reindexed = dfps.reindex(index = newindex)
            dfps_reindexed.interpolate(method = 'linear', inplace = True)
            dfps_reindexed.columns = [str(j)] #(str(i)+str(j))
            dfall = pd.concat([dfall, dfps_reindexed], axis=1)
            
            # plt.plot(dfps, label=str(j))
            # plt.scatter(loc_dt,loc_tp)
            # plt.plot(dfps_reindexed, label=str(i))
            
        # dfall.T.mean()
        # df_ind = pd.concat([df_ind, dfall], axis=1)
        # df_prom = pd.concat([df_prom, dfall.T.mean()], axis=1)
        # df_max = pd.concat([df_max, dfall.T.max()], axis=1)
        # df_min = pd.concat([df_min, dfall.T.min()], axis=1)
    
    events_list = np.arange(0,len(tp_list))
    tp_list = np.asarray(tp_list)
    d_list = np.asarray(d_list)
    sdt_list = np.asarray(sdt_list)
    dt_tpm10_list = np.asarray(dt_tpm10_list)
    tpm10_cum_list = np.asarray(tpm10_cum_list)
    tpm10_list = np.asarray(tpm10_list)
    im10_list = np.asarray(im10_list)
    im60_list = np.asarray(im60_list)
    
    diffs = pd.DataFrame(dt_tpm10_list - sdt_list)[0]
    diff = np.asarray([i.days*24 + i.seconds/3600 for i in diffs])
    
    loc_dt = np.round(diff/d_list,5)
    loc_tp = np.round(tpm10_cum_list/tp_list,5)
    
    for x in loc_dt:
        if x < 0.25:
            loc_dt_quart.append(1)
        elif x < 0.5:
            loc_dt_quart.append(2)
        elif x < 0.75:
            loc_dt_quart.append(3)
        elif x <= 1:
            loc_dt_quart.append(4)

    dfevents = pd.DataFrame({'sdt':sdt_list, 'd':d_list, 'tp':tp_list, 'dt_tpm10':dt_tpm10_list ,'tpm10':tpm10_list, 'im10':im10_list, 'tpm10_cum':tpm10_cum_list, 'im60':im60_list, 'im10_quart':loc_dt_quart}, np.arange(0,len(sdt_list)))
    dfevents.to_csv('./VARIOS/Tables/'+ema_name+'_events.csv')
    
    # diffs = dfevents['dt_tpm10'] - dfevents['sdt']
    # diff = np.asarray([i.days*24 + i.seconds/3600 for i in diffs])
    
    unique, counts = np.unique(loc_dt_quart, return_counts=True)
    dfquart = pd.DataFrame(counts, unique)
    dfquartile = pd.concat([dfquartile, dfquart], axis=1)
    
    # tp_list_norm = tp_list/max(tp_list)
    # d_list_norm = d_list/max(d_list)
    
    ntp_min = 0.5
    ntp_max = 1
    
    nd_min = 0.5
    nd_max = 5

    im10_min = 0
    im10_max = 1
    
    tp_list_norm = (ntp_max-ntp_min)*(tp_list-min(tp_list))/(max(tp_list)-min(tp_list))+ntp_min
    d_list_norm = (nd_max-nd_min)*(d_list-min(d_list))/(max(d_list)-min(d_list))+nd_min
    im10_list_norm = (im10_max-im10_min)*(im10_list-min(im10_list))/(max(im10_list)-min(im10_list))+im10_min
    
    d_list_widths = []
    
    for x in d_list: #<3,6,12,24,>24
        if x < 3:
            d_list_widths.append(0.5)
        elif x < 6:
            d_list_widths.append(2)
        elif x < 12:
            d_list_widths.append(3.5)
        elif x < 24:
            d_list_widths.append(5)
        elif x > 24:
            d_list_widths.append(6.5)

    unique, counts = np.unique(d_list_widths, return_counts=True)
    dfdur = pd.DataFrame(counts, unique)
    dfdurs = pd.concat([dfdurs, dfdur], axis=1)
    
    dur_mean.append(round(d_list.mean(),0))
    
    # len(im10_list)/4
    
    ys = np.repeat([dfall.index.values], len(dfall.columns), axis=0)
    xs = dfall.T.to_numpy()
    
    fig, ax = plt.subplots()
    limits = mpl.colors.Normalize(vmin=0, vmax=round(tp_list.max(),10), clip=False)
    lc = multiline(ys*100, xs*100, c=tp_list, lw=d_list_widths, cmap='Blues', alpha=0.75, norm=limits)
    plt.scatter(loc_dt*100, loc_tp*100, label='Im10', s=8, c='y', alpha=0.75, zorder=2)
    axcb = fig.colorbar(lc, norm=limits)
    axcb.set_label('Total precipitation [mm]')
    ax.set_xlim([0,100])
    ax.set_ylim([0,100])
    ax.set_title('EMA ' + ema_name.capitalize())
    ax.set_xlabel('Cumulative Time [%]')
    ax.set_ylabel('Cumulative Precipitation [%]')
    plt.vlines(25, 0, 100, color='k', ls=':')
    plt.vlines(50, 0, 100, color='k', ls=':')
    plt.vlines(75, 0, 100, color='k', ls=':')
    # plt.plot(dfall.index*100, dfall.median(axis=1)*100, ls='--', label='Median')
    plt.plot(dfall.index*100, dfall.mean(axis=1)*100, lw=2, ls='--', label='Mean', color='g')
    # plt.plot(dfall.index*100, dfall.max(axis=1)*100, lw=2, ls='--', label='Max')
    # plt.plot(dfall.index*100, dfall.min(axis=1)*100, lw=2, ls='--', label='Min')
    x = dfall.quantile(q=[0.1,0.5,0.9], axis=1).T
    lbls = (np.asarray(x.columns)*100).astype(int)
    labels = [str(x)+' %' for x in lbls]
    plt.plot(x.index*100, x*100, lw=2, ls='-.', label=labels, color='r')
    # for col in dfall.columns:
    #     plt.plot(dfall[col], lw=tp_list_norm[int(col)-1], alpha=d_list_norm[int(col)-1], c='k')
    # plt.title('EMA ' + ema_name.capitalize())
    # plt.xlabel('Time')
    # plt.ylabel('Precipitation')
    # plt.plot(dfbaps, ls='--', label='BA')
    # plt.plot([0,1], [0,1], ls='-.')
    ax.legend(ncol=5, fontsize=9, bbox_to_anchor=(1.2, -0.15)) #loc='lower right',
    
    plt.savefig('./VARIOS/Figures/Pcurves/'+ema_name+'.jpg', format='jpg', dpi=1000)
    plt.close()
    
    # x = np.linspace(0,100,6)
    # y = x**0-1+99
    # lwidths = np.linspace(0.5,6.5,5)
    # points = np.array([y, x]).T.reshape(-1, 1, 2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # lc = LineCollection(segments, linewidths=lwidths, color='blue')
    # # fig, ax = plt.subplots()
    # ax.add_collection(lc)
    # # ax.set_xlim(0,5)
    # # ax.set_ylim(0.9,1.1)
    # # fig.show()
    
    dfpall.to_csv('./VARIOS/Tables/'+ema_name+'_p.csv')
    dfall.to_csv('./VARIOS/Tables/'+ema_name+'.csv')
    print('Done', ema_name)


dfdurs = dfdurs.drop(['x'], axis=1)
dfquartile = dfquartile.drop(['x'], axis=1)

dur_group1 = dfdurs.T[0.5]
dur_group2 = dfdurs.T[2.0]
dur_group3 = dfdurs.T[3.5]
dur_group4 = dfdurs.T[5.0]
dur_group5 = dfdurs.T[6.5]
quart1 = dfquartile.T[1]
quart2 = dfquartile.T[2]
quart3 = dfquartile.T[3]
quart4 = dfquartile.T[4]

summary_dur = pd.DataFrame({'dur_mean':dur_mean, 'dur<3':dur_group1, '3<dur<6':dur_group2, '6<dur<12':dur_group3, '12<dur<24':dur_group4, 'dur>24':dur_group5,
                            'quart1':quart1, 'quart2':quart2, 'quart3':quart3, 'quart4':quart4})
summary_dur.to_csv('./VARIOS/Sumamry_dur.csv')






#Plot precipitation

times = [10, 30, 60, 120, 180, 360, 720, 1440]

for ema_event in emas_events:
    print(ema_event)
# for ema_file in emas_files:
    # i += 1
    # ema_file = 'MONTERREY_All.csv'
    # ema_file = emas_files[0]
    # ema_name = ema_file[:-8]
    # folder_name = ema_name+'_Events'
    # ema_event = emas_events[1]
    # ema_event = 'ACAPULCO_Events'
    ema_name = ema_event[:-7]
    ema_file = ema_name+'_All.csv'
    
    # ema = pd.read_csv('./EMAS/'+ema_file) #, parse_dates=['Unnamed: 0'])
    # ema.sort_values(by='Unnamed: 0')
    # ema['Unnamed: 0'].astype(str).sort_values()
    
    # for row, i in ema.iterrows():
    #     if len(i[0])>19:
    #         print(row, i[0], i['P'], len(i[0]))
    
    ema = pd.read_csv('./EMAS/'+ema_file, parse_dates=['Unnamed: 0'])
    ema.index = ema['Unnamed: 0']
    ema = ema.drop(['Unnamed: 0'], axis=1)
    ema.index.name = None
    ema = ema.sort_index()
    
    # ema.iloc[578646]
    # ema[265640:265650]
    # ema.index[ema['P'] == ema['P'].max()]
    # ema['P'].resample('10min').sum()
    
    for t in times:
        # t = 10
        p = ema['P'].resample(str(t)+'min').sum()
    
        plt.plot(p)
        plt.title('EMA '+ ema_name + '-' + str(t) + 'min')
        plt.xlabel('Year')
        plt.ylabel('Cumulative Precipitation - ' + str(t) + ' min [mm]')
        plt.savefig('./VARIOS/Figures/P/'+ema_name+'_'+str(t)+'min.jpg', format='jpg', dpi=1000)
        plt.close()

# ema['P'].max()



i=0
t = 10

i=+1
ema_event = emas_events[i]
ema_name = ema_event[:-7]
ema_file = ema_name+'_All.csv'

ema = pd.read_csv('./EMAS/'+ema_file, parse_dates=['Unnamed: 0'])
ema.index = ema['Unnamed: 0']
ema = ema.drop(['Unnamed: 0'], axis=1)
ema.index.name = None

p = ema['P'] #.resample(str(t)+'min').sum()
pr = p.resample('60min').sum()
prx = pr[207170:-135]

plt.plot(prx)
plt.title('EMA '+ ema_name + '-' + str(t) + 'min')
plt.xlabel('Year')
plt.ylabel('Cumulative Precipitation - ' + str(t) + ' min [mm]')


ema = pd.read_csv('D:/DANI/VARIOS/EMAS/DATOS_ORIGINALES/TODOS/2013/ACAPULCO_2013.csv', parse_dates=['Fecha-Tiempo'])






x = np.linspace(0,5,5)
y = x**0
lwidths=1+x[:-1]
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, linewidths=lwidths, color='b')

fig, ax = plt.subplots()
ax.add_collection(lc)
ax.set_xlim(0,5)
ax.set_ylim(-0.1,0.1)
fig.show()








plt.plot(df_prom)
plt.plot(df_max)
plt.plot(df_min)
plt.plot(dfbaps, ls='--', label='BA')

x = dfall.quantile(q=np.arange(0,1.1,0.1), axis=1).T



idf = [1149.2,0.626] # I = a / D**b
dur = 1440
dt = 60

def altblocks(idf,dur,dt):
    aDur = np.arange(dt,dur+dt,dt)    # in minutes
    aInt = idf[0]/aDur**idf[1]  # idf equation - in mm/h
    # aInt = (idf[0]*RP**idf[1])/((aDur+idf[2])**idf[3])  # idf equation - in mm/h
    aDeltaPmm = np.diff(np.append(0,np.multiply(aInt,aDur/60.0)))
    aOrd = np.append(np.arange(1,len(aDur)+1,2)[::-1],np.arange(2,len(aDur)+1,2))
    prec = np.asarray([aDeltaPmm[x-1] for x in aOrd])
    aAltBl = np.vstack((aDur,prec))
    return aAltBl

idfs = [[2127.8,0.626], [1638.5,0.626], [1491.2,0.626], [1149.2,0.626], [1001.9,0.626], [807.18,0.626], [691.68,0.637], [584.52,0.637], [442.86,0.637]] # I = a / D**b
durs = [180, 360, 720, 1440]
dts = [10,20,30,60]

for idf in idfs:
    for dur in durs:
        for dt in dts:

            blocks = altblocks(idf,dur,dt)
            # plt.bar(blocks[0],blocks[1], width=dt)
            
            dfba_0 = pd.DataFrame([0], index=[0], columns=['P'])
            # dfba_1 = pd.DataFrame([0], index=[blocks[0][-1]+dt], columns=['P'])
            dfba = pd.DataFrame(blocks[1], blocks[0], columns=['P'])
            dfba = pd.concat([dfba_0, dfba]) #, dfba_1])
            dfbas = dfba.cumsum()
            
            Pba = dfba['P'].values
            Pbas = dfbas['P'].values
            Pbasn = np.round(Pbas/Pbas[-1], 5)
            
            tba = dfbas.index.values
            tban = np.round(tba/tba.max(), 5)
            
            dfbaps = pd.DataFrame(Pbasn, tban)
            
            plt.plot(dfbaps, ls='--', label='BA')
            
plt.plot(dfall.T.mean(), ls='--', label='Mean')
plt.plot(dfall.T.max(), ls='--', label='Max')
plt.plot(dfall.T.min(), ls='--', label='Min')


col_names = np.append(dfall.columns.values, ['mean', 'max', 'min'])
dfall2 = pd.concat([dfall, dfall.T.mean(), dfall.T.max(), dfall.T.min()], axis=1)
dfall2.columns = col_names

CN = 85
S = (25400 - 254*CN)/CN
Ia = 0.2*S

Ptotal = 100
dur = 1440
dt = 60
td = np.arange(0,dur+dt,dt)/dur*100

tp = np.arange(0,101,4)
dfall100 = pd.DataFrame(dfall2.values*100, dfall2.index*100)
dfall100.columns = col_names
Pper = dfall100.reindex(index = tp, method='nearest')
Pper['3']

#Storm in percentages
for col in dfall2.columns:
    # print(col)
    # col = '3'
    dfx = dfall2[col]
    tx = dfx.index.values*100
    Px = dfx.values*100
    
    dfpes = pd.DataFrame(Px, tx, columns=['P'])
    Pper = dfpes.reindex(index = td, method='nearest')

#Storm for each event
for col in dfall2.columns:
    # print(col)
    # col = '3'
    
    dfx = dfall2[col]
    tx = dfx.index.values
    Px = dfx.values
    
    Pac = Px*Ptotal
    t = tx*dur
    
    dfpes = pd.DataFrame(Pac, t, columns=['P'])
    Paci = dfpes.reindex(index = td, method='nearest')
    Pac = Paci['P'].values
    
    # Pe = (P-Ia)**2/(P-Ia+S)
    Pe = [0 if x < Ia else (x-Ia)**2/(x-Ia+S) for x in Pac]
    
    dfpes = pd.DataFrame(Pe, td, columns=['P'])
    dfpe = dfpes.diff()
    dfpe.iloc[0] = 0
    plt.plot(dfpe)
    # plt.bar(dfpe.index, dfpe['P'].values, width=dt-5)

#Alternatig blocks method
Pac = Pbasn*Ptotal
t = tban*dur

dfpes = pd.DataFrame(Pac, t, columns=['P'])
Paci = dfpes.reindex(index = td, method='nearest')
Pac = Paci['P'].values

# Pe = (P-Ia)**2/(P-Ia+S)
Pe = [0 if x < Ia else (x-Ia)**2/(x-Ia+S) for x in Pac]

dfpes = pd.DataFrame(Pe, td, columns=['P'])
dfpe = dfpes.diff()
dfpe.iloc[0] = 0
plt.plot(dfpe)
plt.bar(dfpe.index, dfpe['P'].values, width=dt-5)





