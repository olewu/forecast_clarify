# %load_ext autoreload
# %autoreload 2

from pyclarify import ClarifyClient, query, SignalInfo, DataFrame
import matplotlib.pyplot as plt
from forecast_clarify.main import *


client = ClarifyClient('../data/credentials/clarify-credentials_SLS.json')

response = client.select_items(include_dataframe=False)
item_dict = response.result.items
for item_id, meta_data in item_dict.items():
  print(f"ID: {item_id} \t Name: {meta_data.name}")
  
  
item_id = ["ca8dhakpllnee2k58qfg"]

filter = query.Filter(fields={"id": query.In(value=item_id)})
response_unfiltered = client.select_items(
    filter = filter,
    include_metadata = True,
    include_dataframe = True,
    not_before = "2022-07-01T00:00:00Z",
    before = datetime.today()
)
print(response_unfiltered.result.data.times[-1])
# the data (at least for this item) is available with a lag of only two hours


response_filtered = client.select_items(
    filter = filter,
    include_metadata = True,
    include_dataframe = True,
    not_before = "2019-05-01T00:00:00Z",
    before = datetime.today(),
    rollup = "P7DT"
)
# question: is the time tag that comes out of the rollup the center of the window?

st3m_langoey = response_filtered.result.data.to_pandas()
print(st3m_langoey.index[-10:])



response_filtered_1D = client.select_items(
    filter = filter,
    include_metadata = True,
    include_dataframe = True,
    not_before = "2019-05-01T00:00:00Z",
    before = datetime.today(),
    rollup = "P1DT"
)
# question: is the time tag that comes out of the rollup the center of the window?

st3m_langoey_daily = response_filtered_1D.result.data.to_pandas()
print(st3m_langoey_daily.index[-10:])



f,ax = plt.subplots(figsize=(10,4))
st3m_langoey.ca8dhakpllnee2k58qfg_avg.plot(ax=ax)
st3m_langoey.ca8dhakpllnee2k58qfg_min.plot(ax=ax)
st3m_langoey.ca8dhakpllnee2k58qfg_max.plot(ax=ax)
st3m_langoey_daily.ca8dhakpllnee2k58qfg_avg.plot(ax=ax,ls='dotted')
ax.grid()
ax.set_ylabel('{0:s}  {1:s} /{2:s}'.format(
        response_unfiltered.result.items[item_id[0]].name,
        response_unfiltered.result.items[item_id[0]].labels['depth'][0],
        response_unfiltered.result.items[item_id[0]].engUnit
    )
);
ax.set_title('{0:s} {1:s} 7-day average'.format(
        response_unfiltered.result.items[item_id[0]].name,
        response_unfiltered.result.items[item_id[0]].labels['site'][0]
    )
);


plt.savefig('fig1.png')



SC = seas_cycle(st3m_langoey.ca8dhakpllnee2k58qfg_avg,nharm=3)
SC.fit()
SC.training_anomalies()

SC_1 = seas_cycle(st3m_langoey_daily.ca8dhakpllnee2k58qfg_avg)
SC_1.fit()
SC_1.training_anomalies()

SC_varharm = seas_cycle(st3m_langoey.ca8dhakpllnee2k58qfg_avg,nharm=1)
SC_varharm.fit()
SC_varharm.training_anomalies()



f,ax = plt.subplots(figsize=(10,5))

SC.sc_exp_doy.ca8dhakpllnee2k58qfg_avg.plot(ax=ax,color='C1')
SC.absolute_vals.ca8dhakpllnee2k58qfg_avg.plot(ax=ax,color='C0')
SC.anomalies.ca8dhakpllnee2k58qfg_avg.plot(ax=ax,color='C2')

SC_varharm.sc_exp_doy.ca8dhakpllnee2k58qfg_avg.plot(ax=ax,ls='dotted',color='C1')
# SC_varharm.absolute_vals.ca8dhakpllnee2k58qfg_avg.plot(ax=ax,ls='dotted')
SC_varharm.anomalies.ca8dhakpllnee2k58qfg_avg.plot(ax=ax,ls='dotted',color='C2')

ax.grid()
plt.savefig('fig2.png')

f,ax = plt.subplots(figsize=(10,5))
SC.absolute_vals.ca8dhakpllnee2k58qfg_avg.plot(x='month_day',ax=ax)
SC.abs_doy_mean.ca8dhakpllnee2k58qfg_avg.plot(ax=ax)
SC.mean_sc.ca8dhakpllnee2k58qfg_avg.plot(ax=ax)
# SC_1.absolute_vals.ca8dhakpllnee2k58qfg_avg.plot(x='month_day',ax=ax)
# SC_1.abs_doy_mean.ca8dhakpllnee2k58qfg_avg.plot(ax=ax)
# SC_1.mean_sc.ca8dhakpllnee2k58qfg_avg.plot(ax=ax)


anom_pers = persistence(lags=4)
anom_pers.fit(SC.anomalies.ca8dhakpllnee2k58qfg_avg)


# prediction of the anomaly:
anom_pred = anom_pers.predict(SC.anomalies.ca8dhakpllnee2k58qfg_avg.isel(time=-1))

# prediction of seasonal cycle:
sc_pred = SC.predict(anom_pred.time_doy,time_name='lags')

abs_pred = anom_pred + sc_pred


f,ax = plt.subplots(figsize=(10,5))

SC.sc_exp_doy.ca8dhakpllnee2k58qfg_avg.plot(ax=ax,color='C1')
SC.absolute_vals.ca8dhakpllnee2k58qfg_avg.plot(ax=ax,color='C0')
SC.anomalies.ca8dhakpllnee2k58qfg_avg.plot(ax=ax,color='C2')

anom_pred.plot(ax=ax,x='time',color='C3')
sc_pred.ca8dhakpllnee2k58qfg_avg.plot(ax=ax,x='time',color='C4',ls='dotted')
abs_pred.ca8dhakpllnee2k58qfg_avg.plot(ax=ax,x='time',color='C3')

ax.grid()
plt.savefig('fig3.png')

signal_name = '{0:s} SS_forecast'.format(response_filtered.result.items[item_id[0]].name)
signal_desc = 'Persistence forecast of 7-day averages up to 4 weeks ahead'
signal_unit = response_filtered.result.items[item_id[0]].engUnit
signal_labels = {'data-source':['Persistence Model'],'site':response_filtered.result.items[item_id[0]].labels['site'],'depth':response_filtered.result.items[item_id[0]].labels['depth']}




# put forecast values into list:
fc_list = list(abs_pred.ca8dhakpllnee2k58qfg_avg.values)

# make list with corresponding dates:
time_list = list(abs_pred.time.values)



# Create a signal and write metadata to it
signal = SignalInfo(name = signal_name, description = signal_desc, engUnit = signal_unit, labels = signal_labels, sourceType = 'prediction')
client.save_signals(input_ids=['persistence_fc_test'], signals=[signal], create_only=False)
# Write data into a signal
data = DataFrame(series={'persistence_fc_test': fc_list}, times = time_list)
client.insert(data)


### Merge obs with forecast

# write signals back to clarify
signal_name = '{0:s} observations_forecast'.format(response_filtered.result.items[item_id[0]].name)
signal_desc = 'Observations and Persistence forecast of 7-day averages up to 4 weeks ahead'
signal_unit = response_filtered.result.items[item_id[0]].engUnit
signal_labels = {'data-source':['Clarify obs + Persistence Model'],'site':response_filtered.result.items[item_id[0]].labels['site'],'depth':response_filtered.result.items[item_id[0]].labels['depth']}


a1 = st3m_langoey.ca8dhakpllnee2k58qfg_avg.rename("t3m")
a2= abs_pred.ca8dhakpllnee2k58qfg_avg.drop('time_doy').to_dataframe().reset_index(level=0,drop=True).set_index('time').rename(columns={"ca8dhakpllnee2k58qfg_avg": "t3m"})

# put forecast values into list:
obs_fc_list = list(np.concatenate((a1, a2), axis=None))

# make list with corresponding dates:
obs_fc_time_list = list(np.concatenate((a1.index.values, a2.index.values), axis=None))

# Create a signal and write metadata to it
signal = SignalInfo(name = signal_name, description = signal_desc, engUnit = signal_unit, labels = signal_labels, sourceType = 'prediction')
client.save_signals(input_ids=['obs_persistence_fc'], signals=[signal], create_only=False)
# Write data into a signal
data_obs_fc = DataFrame(series={'obs_persistence_fc': obs_fc_list}, times = obs_fc_time_list)
client.insert(data_obs_fc)


