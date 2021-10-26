import numpy as np
import pandas as pd
from fbprophet import Prophet

d1 = [5008610, 9062190, 4682900, 4802795, 5817090, 4098160, 4807535, 9320945, 5349790, 4214450, 4045375, 5640105]
d2 = [5945585, 9402190, 4970611, 4683556, 5856100, 4073460, 4472560, 8054250, 8708860, 5734100, 5044805, 8399930]
d3 = [8558210, 0,       1984425, 5220560, 8880910, 6678150, 6780305, 13572558, 7883881, 5696470, 4273230, 8921500]
d4 = [4819400, 10194460, 6002033, 5108700, 6390200, 4633747, 5547380, 10657670]

d = d1 + d2 + d3 + d4
print('mean', np.mean(d))
print('sum', np.sum(d))
print('std', np.std(d))
print('19vs18:', (np.mean(d2) - np.mean(d1)) / np.mean(d1))
print('20vs19:', (np.mean(d3) - np.mean(d2)) / np.mean(d2))
print('21vs20:', (np.mean(d4) - np.mean(d3)) / np.mean(d3))

## -----quater ----##
import torch
import torch.nn as nn

d_matrix = np.array([np.array(d1), np.array(d2), np.array(d3), np.array(d4)])
print(d_matrix)

dates = pd.date_range('20180101', periods=44, freq='M')
print(dates)

data = np.reshape(np.array(d), [len(d), 1])
timesteps = np.reshape(np.arange(len(d)), [len(d), 1])
# all_data = np.concatenate([dates, data], axis=1)

df = {'ds':pd.Series(dates) }
df = pd.DataFrame(df)

# df = pd.DataFrame(dates)# , columns={'timestamp':'ds', 'value':'y'})
df['y'] = data
#
# #df['dates'] = dates
# #df0 = pd.DataFrame(dates)
# df = df.rename(columns={'0':'ds', 'value':'y'})

# df['y'] = (df['y'] - df['y'].mean()) / (df['y'].std())

print(df)

m = Prophet(yearly_seasonality=True)
m.fit(df)
future = m.make_future_dataframe(periods=30, freq='min')
future.tail()
forecast = m.predict(future)
print(forecast.tail())

#m.plot(forecast)
#m.plot_components(forecast)
#m.add_seasonality(name='monthly', period=30.5, fourier_order=5)





