import pandas as pd
# load data from csv
df = pd.read_csv('nsadata/logins.txt', sep='\t', header=None, names=['date', 'time', 'user', 'way'])
# People have normal schedules of when they like to show up to work.

# Find an account that starts with j. of someone who sometimes works a different shift than their normal.

# we can do this by grouping by user, getting median time, and then finding the user who has a different time
# than the median
# get second of day
df['time_second'] = pd.to_datetime(df['time']).dt.hour * 3600 + pd.to_datetime(df['time']).dt.minute * 60 + pd.to_datetime(df['time']).dt.second
df_new = df[df['way']=='IN'].groupby('user')['time_second'].apply(lambda x: x.median())#.reset_index(name='median_time')


df_in = df[(df['way']=='IN') & (df['user'].str.startswith('j.'))] 
# add a column for distance from median and then sort and get highest
df_in['time_diff'] = abs(df_in['time_second'] - df_in['time_second'].median())

# calculate standard deviation for every user and then find the user with the highest standard deviation
print(df_in.groupby('user')['time_second'].std().sort_values(ascending=False).index[0])
# 'j.salano'