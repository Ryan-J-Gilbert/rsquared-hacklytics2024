import pandas as pd
# load data from csv
df = pd.read_csv('nsadata/logins.txt', sep='\t', header=None, names=['date', 'time', 'user', 'way'])
unique = df[df['way'] == 'IN']['user'].nunique()

# how many people's first traffic was 'OUT'
# group by user and get first way
df_first = df.groupby('user').first()
logout = df_first[df_first['way'] == 'OUT'].shape[0]
print(unique-logout)
# 5394