import pandas as pd
# load data from csv
df = pd.read_csv('nsadata/logins.txt', sep='\t', header=None, names=['date', 'time', 'user', 'way'])
# maybe they signed in around the same time with some variance which changed after July

# split data into before and after July
df['date'] = pd.to_datetime(df['date'])
df['time'] = pd.to_datetime(df['time'])
df['seconds'] = df['time'].dt.hour * 3600 + df['time'].dt.minute * 60 + df['time'].dt.second
df_before = df[df['date'] < '2021-07-01']
df_after = df[df['date'] >= '2021-07-01']

# group by user and get std way=IN time
df_before_in = df_before[df_before['way'] == 'IN'].groupby('user')['seconds'].std()
df_after_in = df_after[df_after['way'] == 'IN'].groupby('user')['seconds'].std()

# find user with the highest change in std after July
(df_after_in).sort_values(ascending=True).index[0]
# turns out maybe just using before july? the std is 0
# d.tye