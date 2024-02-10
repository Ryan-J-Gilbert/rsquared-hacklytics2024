import pandas as pd
# load data from csv
df = pd.read_csv('nsadata/logins.txt', sep='\t', header=None, names=['date', 'time', 'user', 'way'])
# In this dataset, traffic is simulated in 5 zones of the city: north, east, south, west, and central. User s.kinkel lives in the west zone.

# **Which day of the year, in YYYY-MM-DD format, did the west zone have the worst traffic? **

# Note that s.kinkel may not have worked that day.
# we can groupby date and find the count and then sort by count descending
df_west = df[df['user'] != 's.kinkel']
df_west['date'] = pd.to_datetime(df_west['date'])
df_west.groupby('date').size().sort_values(ascending=False).index[0]
# 2021-01-20