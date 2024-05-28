"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import numpy as np
from numpy import percentile
import datetime as dt
import matplotlib.pyplot as plt
plt.style.use({'figure.facecolor':'white'})
from sklearn.preprocessing import StandardScaler
from pyod.models.hbos import HBOS
from pyod.models.abod import ABOD
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.pca import PCA
from pyod.models.iforest import IForest
from pyod.models.lscp import LSCP

import time
#st.set_page_config(layout="wide")

st.title("Анализ аномалий")

uploaded_file = st.file_uploader("Загрузите файл датасета...", accept_multiple_files=False)
useDemoCSV = st.checkbox('Использовать демонстрационный датасет')
#параметры моделирования
outliers_fraction= 0.05
method= 'LSCP'
models= ['HBOS', 'ABOD', 'LOF', 'CBLOF', 'PCA', 'IForest']
ensemble_func='аномалия отмечена всеми моделями' 
lof_neighbors=20
#отображать или нет параметры
show_params= st.checkbox('Показать параметры')
if show_params:
    outliers_fraction = st.slider(
        "Доля аномальных значений",
        0.0, 0.5, 0.05)
    method = st.selectbox(
        'Использовать метод поиска аномалий',
         ['LSCP', 'Ансамбль моделей (простой)'] + models)
    if method=='Ансамбль моделей (простой)':
        ensemble_func = st.selectbox(
        'Отметить значение как аномальное, если',
         ['аномалия отмечена всеми моделями', 'аномалия отмечена большей частью моделей', 'аномалия отмечена любой из моделей'])
    if method=='LOF' or method=='Ансамбль моделей (простой)' or method=='LSCP':
        lof_neighbors= st.number_input('Параметр "число соседей" (neighbors) модели LOF', value= lof_neighbors)
if uploaded_file or useDemoCSV:
    if uploaded_file:
        file_path= uploaded_file.name
    else:
        file_path= 'dataset.csv'
    df= pd.read_csv(file_path, parse_dates=['datetime'])
    st.header('Данные датасета:')
    st.write(df.head())

    def show_plots(df, columns, anomaly_column=None):
        fig, axs = plt.subplots(len(columns), 1, sharex=True, constrained_layout=True, figsize=(8,6))
        for i in range(len(columns)):
            c = columns[i]

            axs[i].plot(dd.index, df[c], color='gray',label='Normal')

            if anomaly_column:
                a = df.loc[df[anomaly_column] == 1, [c]] #anomaly
                axs[i].scatter(a.index, a[c], color='red', label='Anomaly')

            axs[i].xaxis_date()
            axs[i].set_title(c)
            plt.xlabel('Date')
        st.pyplot(fig, use_container_width = True)
        
    method_cols= []
    def get_amonaly_decision(df_raw, ensemble_func='аномалия отмечена всеми моделями'):
        cnt= 0 #количество предсказаний о наличии аномалии
        cnt_all= len(method_cols)#всего моделей
        for col in method_cols:
            if(df_raw[col]==1):
                cnt=cnt+1
        anomaly_decision= 0 #по умолчанию аномалии нет
        if ensemble_func=='аномалия отмечена всеми моделями': 
            if cnt==cnt_all: 
                anomaly_decision= 1
        elif ensemble_func=='аномалия отмечена большей частью моделей': 
            if cnt/cnt_all>0.5: 
                anomaly_decision= 1
        else: #ensemble_func=='аномалия отмечена любой из моделей' 
           if cnt>0: 
                anomaly_decision= 1     
        return anomaly_decision

    method_title= method
    if method=='Ансамбль моделей (простой)' or method=='LSCP':
        method_title+= ' ('
        for model in models:
            if method=='LSCP' and model=='ABOD':
                continue
            method_title+= model
            method_title+= ', '
        method_title = method_title.rstrip(', ')
        method_title+= ')'
            
    st.header(method_title)
    columns = ['rto', 'cheques', 'n_sku', 'cnt', 'cashnum']
    random_state = np.random.RandomState(42)
    detectors=[]
    if method=='Ансамбль моделей (простой)' or method=='LSCP':
        detectors_list=[]
        detectors_list.append(HBOS(contamination=outliers_fraction))
        if method!='LSCP':
            detectors_list.append(ABOD(contamination=outliers_fraction))
        detectors_list.append(LOF(n_neighbors=lof_neighbors, contamination=outliers_fraction))
        detectors_list.append(CBLOF(contamination=outliers_fraction, random_state=random_state))
        detectors_list.append(PCA(contamination=outliers_fraction, random_state=random_state))
        detectors_list.append(IForest(contamination=outliers_fraction, random_state=random_state))
        if method=='LSCP':
            detectors.append((LSCP(detectors_list, contamination=outliers_fraction, random_state=random_state), 'LSCP'))
        else:
            for detector, model in zip(detectors_list, models):
                detectors.append((detector, model)) 
    elif method=='HBOS':
        detectors.append((HBOS(contamination=outliers_fraction), 'HBOS'))
    elif method=='ABOD':
        detectors.append((ABOD(contamination=outliers_fraction), 'ABOD'))
    elif method=='LOF':
        detectors.append((LOF(n_neighbors=lof_neighbors, contamination=outliers_fraction), 'LOF'))
    elif method=='CBLOF':
        detectors.append((CBLOF(contamination=outliers_fraction, random_state=random_state), 'CBLOF'))
    elif method=='PCA':
        detectors.append((PCA(contamination=outliers_fraction, random_state=random_state), 'PCA'))
    else:# method=='IForest':
        detectors.append((IForest(contamination=outliers_fraction, random_state=random_state), 'IForest'))
    latest_iteration = st.empty()
    latest_iteration.text(f'Поиск аномалий и вывод диаграмм')
    bar = st.progress(0)
    i=0
    step= 1/len(df.ou.unique())
    cnt=1
    for ou in df.ou.unique():
        st.write(f'Идентификатор магазина: {int(ou)}')
        dd = df[df.ou==ou]
        dd = dd.set_index('datetime')
        dd = dd[columns].resample('D').sum()

        X = dd.values
        
        for detector in detectors:
            detector[0].fit(X)
            pred = detector[0].predict(X)
            col= detector[1]
            dd[col]= pred
            method_cols.append(col)

        dd['anomaly'] = dd.apply(lambda x: get_amonaly_decision(x, ensemble_func), axis=1)
        show_plots(dd, columns, 'anomaly')
        latest_iteration.text(f'Поиск аномалий и вывод диаграмм для {cnt} из {len(df.ou.unique())}')
        bar.progress(i+step)
        i=i+step
        cnt=cnt+1
    latest_iteration.text(f'Готово!')
    time.sleep(3)
    latest_iteration.text(f'')