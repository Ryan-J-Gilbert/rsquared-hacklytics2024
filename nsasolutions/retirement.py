import pandas as pd
# load data from csv
df = pd.read_csv('nsadata/logins.txt', sep='\t', header=None, names=['date', 'time', 'user', 'way'])
# group by user then compare the difference in day (retrieved from date) between the next row
# then find row where the difference is greater than 100
df_ms = df[df['user'].str.startswith('m.')]
df_ms['date'] = pd.to_datetime(df_ms['date'])
df_ms['date_diff'] = df_ms.groupby('user')['date'].diff().dt.days
print(df_ms[df_ms['date_diff'] > 20]['user'].iloc[0])
# m.ponds