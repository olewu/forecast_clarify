# train the persistence model on Norkyst800 data

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from forecast_clarify.main import *
from forecast_clarify.config import *

# load training data:
ds = xr.open_dataset('/projects/NS9853K/DATA/norkyst800/station_3m/temperature_insitu_3m_norkyst800_barentswatch_closest_20060101-20220920.nc')

filepath = dirs['param_files']

#--------------1. Estimate trend--------------#
# initialize trend class w/ desired polynomial degree:
trnd = trend(degree=1)
# train trend model:
trnd.fit(ds.squeeze())
# save trend to file:
trnd.save(os.path.join(filepath,'t3m_trend_nk800_2006-2022_ff.nc'))


#--------------2. Estimate seasonal cycle of the detrended data--------------#
# detrend
data_dtr = trnd.detrend()

# train mean seasonal cycle:
SC = seas_cycle(data_dtr,nharm=3)
SC.fit()
SC.save(os.path.join(filepath,'t3m_seasonal_cycle_nk800_2006-2022_ff.nc'))
# create anomalies of the training data:
SC.training_anomalies()

# train standard deviation seasonal cycle:
SC_std = seas_cycle(SC.anomalies,nharm=3,moment='std')
SC_std.fit()
SC_std.training_anomalies()
SC_std.save(os.path.join(filepath,'t3m_seasonal_cycle_std_nk800_2006-2022_ff.nc'))


#--------------3. Create anomalies and estimate lagged (running) correlation--------------#
# process to weekly means:
anoms_weekly = SC_std.anomalies.resample(time='7D').mean('time')

# train persistence model
pers = persistence(lags=5)
pers.fit(anoms_weekly)
pers.save(os.path.join(filepath,'t3m_persistence_nk800_2006-2022_ff.nc'))