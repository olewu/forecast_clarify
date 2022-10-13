# train the persistence model on Norkyst800 data

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from forecast_clarify.main import *
from forecast_clarify.config import *


def train_persistence_model_weekly(training_file,trend_deg=1,standardize=True,seas_harm=3,lags=5,seasonal_pers=True,wndw=61,pers_harmonics=None):
    
    # load training data:
    ds = xr.open_dataset(training_file)

    fname = training_file.split('/')[-1].split('.')[0]

    #--------------1. Estimate trend--------------#
    # initialize trend class w/ desired polynomial degree:
    trnd = trend(degree = trend_deg)
    # train trend model:
    trnd.fit(ds.squeeze())
    # save trend to file:
    trnd.save(os.path.join(dirs['param_files'],fname+'_trend.nc'))


    #--------------2. Estimate seasonal cycle of the detrended data--------------#
    # detrend
    data_dtr = trnd.detrend()

    # train mean seasonal cycle:
    SC = seas_cycle(data_dtr,nharm=seas_harm)
    SC.fit()
    SC.save(os.path.join(dirs['param_files'],fname+'_seasonal_cycle.nc'))
    # create anomalies of the training data:
    SC.training_anomalies()

    if standardize:
        # train standard deviation seasonal cycle:
        SC_std = seas_cycle(SC.anomalies,nharm=seas_harm,moment='std')
        SC_std.fit()
        SC_std.training_anomalies()
        SC_std.save(os.path.join(dirs['param_files'],fname+'_seasonal_cycle_std.nc'))

    else:
        SC_std = SC

    #--------------3. Create anomalies and estimate lagged (running) correlation--------------#
    # process to weekly means:
    anoms_weekly = SC_std.anomalies.resample(time='7D').mean('time')

    if not seasonal_pers:
        # train persistence model
        pers = persistence(lags=lags)
        pers.fit(anoms_weekly)
        pers.save(os.path.join(dirs['param_files'],fname+'_persistence.nc'))
    else:
        pers = persistence_seasonal(lags=lags,wndw=wndw,harmonics=pers_harmonics)
        pers.fit(anoms_weekly)
        pers.save(os.path.join(dirs['param_files'],fname+'_persistence_seasonal_cycle_{0:d}D-wndw_{1:d}harm.nc'.format(wndw,pers_harmonics)))

if __name__ == '__main__':
    training_file = '/projects/NS9853K/DATA/norkyst800/station_3m/temperature_insitu_3m_norkyst800_barentswatch_closest_20060101-20220920.nc'
    train_persistence_model_weekly(training_file,pers_harmonics=3)