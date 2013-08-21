import sqlite3
import numpy as np
import pandas as pd
con=sqlite3.connect('pl.db')



cur = con.cursor() 
#query =('SELECT * FROM %s') %table
query =('SELECT * FROM hf_general')
cur.execute(query) 
hf_general=cur.fetchall()
cur.execute('PRAGMA table_info(hf_general);')
columns=cur.fetchall()
columns=np.array(columns)
colnames=columns[:,1]
hf_general=pd.DataFrame(hf_general,columns=colnames)




query =('SELECT * FROM hf_income')
cur.execute(query) 
hf_income=cur.fetchall()
cur.execute('PRAGMA table_info(hf_income);')
columns=cur.fetchall()
columns=np.array(columns)
colnames=columns[:,1]
hf_income=pd.DataFrame(hf_income,columns=colnames)





query =('SELECT * FROM hf_weather')
cur.execute(query) 
hf_weather=cur.fetchall()
cur.execute('PRAGMA table_info(hf_weather);')
columns=cur.fetchall()
columns=np.array(columns)
colnames=columns[:,1]
hf_weather=pd.DataFrame(hf_weather,columns=colnames)



query =('SELECT * FROM hf_population')
cur.execute(query) 
hf_population=cur.fetchall()
cur.execute('PRAGMA table_info(hf_population);')
columns=cur.fetchall()
columns=np.array(columns)
colnames=columns[:,1]
hf_population=pd.DataFrame(hf_population,columns=colnames)




#merging dfs
data=hf_general.merge(hf_income,on='zip')
data=data.merge(hf_weather,on='zip')
data=data.merge(hf_population,on='zip')

#saving to csv
data.to_csv('homefair.csv',encoding='utf-8')
