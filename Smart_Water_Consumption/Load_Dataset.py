import pandas as pd
import numpy as np
from datetime import datetime, timedelta

n_meters = 100
hours = 24 * 30
timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours)][::-1]

data = {
    'timestamp': np.tile(timestamps, n_meters),
    'meter_id': np.repeat(range(1, n_meters + 1), hours),
    'flow_rate': np.random.uniform(50, 500, hours * n_meters),
    'location': np.repeat(['Zone A', 'Zone B', 'Zone C', 'Zone D'], hours * n_meters // 4)
}

df = pd.DataFrame(data)

df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

df
