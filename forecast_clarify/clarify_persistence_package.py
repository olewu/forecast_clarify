
from forecast_clarify.main import *
from forecast_clarify.config import *
import os
import json

def find_station_in_bw(station_name,return_latlon=False):

    with open(bw_sites_file) as f: # location defined in config
        data = json.load(f)

    stat_specs = [(d['name'],d['localityNo'],d['lat'],d['lon']) for d in data if station_name == d['name']]
    ex = 'exact'
    if not stat_specs:
        print('No exact match found, searching for similar station names')
        stat_specs = [(d['name'],d['localityNo'],d['lat'],d['lon']) for d in data if station_name in d['name']]
        ex = 'inexact'

    if len(stat_specs) > 1:
        dupls = '\n\t'.join([sp[0] for sp in stat_specs])
        print('WARNING: {0:} {1:s} matches for requested station.\nOptions are:\n\t{2:s}'.format(len(stat_specs),ex,dupls))

    stat_id = [sp[1] for sp in stat_specs]
    stat_lat = [sp[2] for sp in stat_specs]
    stat_lon = [sp[3] for sp in stat_specs]

    if return_latlon:
        return stat_id,stat_lat,stat_lon
    else:
        return stat_id


def make_persistence_forecast(init_value,init_time,station_id=None,standardized_pers=True,as_dataframe=True):
    """
    assumes weekly means
    """
    
    # collect model parameters:
    model = collect_model()

    nlag = model['persistence'].lags + 1 # +1 to return initialization value

    t0_xr = xr.DataArray(
        pd.date_range(start=init_time,freq='7D',periods=nlag),
        dims = {'lags':nlag},
        coords = {
            'lags':('lags',np.arange(6)),
            'time':('lags',pd.date_range(start=init_time,freq='7D',periods=nlag))
        }
    )
    DOY = get_doy_coord(t0_xr).assign_coords({'lags':('time',np.arange(6))}).swap_dims({'time':'lags'})

    # predict climatological component:
    trend_pred = model['trend'].predict(t0_xr)
    SC_pred = model['seasonal_cycle_mean'].predict(DOY,time_name='lags')

    if standardized_pers:
        # seasonal cycle of std:
        SC_std_pred = model['seasonal_cycle_std'].predict(DOY,time_name='lags')
    else:
        # dummy full of ones
        SC_std_pred = xr.DataArray(np.ones_like(SC_pred),dims=SC_pred.dims,coords=SC_pred.coords)
    
    # form initial anomaly as deviation from above obtained climatology 
    init_anom = (init_value - (SC_pred + trend_pred - model['trend'].mean).sel(lags=0))/SC_std_pred.sel(lags=0)
    init_anom = init_anom.assign_coords({'month_day':DOY[0]})

    # predict anomalies:
    anom_fc = model['persistence'].predict(init_anom)

    # add anomalies back to climatology as predicted for the desired dates above:
    abs_temp_fc = anom_fc*SC_std_pred + (SC_pred + trend_pred - model['trend'].mean)
    abs_temp_fc.name = 'temperature'

    if station_id is not None:
        abs_temp_fc = abs_temp_fc.sel(location=station_id)

    if not as_dataframe:
        return abs_temp_fc 
    else:
        return abs_temp_fc.to_dataframe()


def collect_model(components=('trend','seasonal_cycle_mean','seasonal_cycle_std','persistence')):
    
    components_coll = {}
    # trend:
    if 'trend' in components:
        trnd = trend()
        trnd.load_trend(os.path.join(dirs['param_files'],'t3m_trend_nk800_2006-2022_ff.nc'))
        components_coll['trend'] = trnd
    # seasonal cycle of mean:
    if 'seasonal_cycle_mean' in components:
        SC = seas_cycle(1,load_mode=True)
        SC.load_sc(os.path.join(dirs['param_files'],'t3m_seasonal_cycle_nk800_2006-2022_ff.nc'))
        components_coll['seasonal_cycle_mean'] = SC
    # seasonal cycle of std:
    if 'seasonal_cycle_std' in components:
        SC_std = seas_cycle(1,load_mode=True)
        SC_std.load_sc(os.path.join(dirs['param_files'],'t3m_seasonal_cycle_std_nk800_2006-2022_ff.nc'))
        components_coll['seasonal_cycle_std'] = SC_std
    # persistence:
    if 'persistence' in components:
        pers = persistence()
        pers.load(os.path.join(dirs['param_files'],'t3m_persistence_nk800_2006-2022_ff.nc'))
        components_coll['persistence'] = pers
    
    return components_coll