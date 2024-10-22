df['timestamp'] = pd.to_datetime(df['timestamp'])

print(df.isnull().sum())

df['flow_rate'].fillna(df['flow_rate'].mean(), inplace=True)

df = df[df['flow_rate'] <= 1000]
