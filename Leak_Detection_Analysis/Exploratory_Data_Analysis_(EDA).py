import matplotlib.pyplot as plt
import seaborn as sns

meter_id = 1
meter_data = df[df['meter_id'] == meter_id]

plt.figure(figsize=(10, 6))
plt.plot(meter_data['timestamp'], meter_data['flow_rate'])
plt.title(f'Water Consumption for Meter {meter_id}')
plt.xlabel('Timestamp')
plt.ylabel('Flow Rate (Liters per Hour)')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='location', y='flow_rate', data=df)
plt.title('Water Flow Rate Distribution by Location')
plt.show()
