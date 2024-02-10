import pandas as pd
# load data from csv
df = pd.read_csv('nsadata/logins.txt', sep='\t', header=None, names=['date', 'time', 'user', 'way'])

# need to find entries that have multiple logins (WAY = "IN") in the same day
# group by date and user, and count the number of logins greater than 1
df_new = df[df['way'] == 'IN'].groupby(['date', 'user']).size().reset_index(name='count')

# count greater than 1 and user stars with 'e.'
print(df_new[(df_new['count'] > 1) & (df_new['user'].str.startswith('e.'))]['user'].iloc[0])

# e.17