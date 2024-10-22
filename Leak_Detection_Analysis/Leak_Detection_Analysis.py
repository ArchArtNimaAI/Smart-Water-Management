leak_threshold = 400
df['leak_detected'] = df['flow_rate'] > leak_threshold

leak_counts = df[df['leak_detected']].groupby('meter_id').size().reset_index(name='leak_count')
df
print(leak_counts.sort_values(by='leak_count', ascending=False))
