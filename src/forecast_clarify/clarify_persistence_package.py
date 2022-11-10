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
    
    # collect model parameters for persistence model with seasonally varying persistence:
    model = collect_model(wndw=61,harm=3)

    nlag = model['persistence'].lags + 1 # +1 to return initialization value

    t0_xr = xr.DataArray(
        pd.date_range(start=init_time,freq='7D',periods=nlag),
        dims = {'lags':nlag},
        coords = {
            'lags':('lags',np.arange(nlag)),
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

    # correct the doy coordinate to range from 1 to 365:
    abs_temp_fc = abs_temp_fc.assign_coords(time_doy=(abs_temp_fc.time_doy-1)%365 + 1)

    if station_id is not None:
        abs_temp_fc = abs_temp_fc.sel(location=station_id)

    if not as_dataframe:
        return abs_temp_fc 
    else:
        return abs_temp_fc.to_dataframe()


def collect_model(components=('trend','seasonal_cycle_mean','seasonal_cycle_std','persistence'),filename_base = 'temperature_insitu_3m_norkyst800_barentswatch_closest_20060101-20220920_',wndw=None,harm=None):
    
    components_coll = {}
    # trend:
    if 'trend' in components:
        trnd = Trend()
        trnd.load_trend(os.path.join(dirs['param_files'],filename_base + 'trend.nc'))
        components_coll['trend'] = trnd
    # seasonal cycle of mean:
    if 'seasonal_cycle_mean' in components:
        SC = SeasonalCycle(1,load_mode=True)
        SC.load_sc(os.path.join(dirs['param_files'],filename_base + 'seasonal_cycle.nc'))
        components_coll['seasonal_cycle_mean'] = SC
    # seasonal cycle of std:
    if 'seasonal_cycle_std' in components:
        SC_std = SeasonalCycle(1,load_mode=True)
        SC_std.load_sc(os.path.join(dirs['param_files'],filename_base + 'seasonal_cycle_std.nc'))
        components_coll['seasonal_cycle_std'] = SC_std
    # persistence:
    if 'persistence' in components:
        if wndw is None:
            pers = Persistence()
            pers.load(os.path.join(dirs['param_files'],filename_base + 'persistence.nc'))
        # seasonally varying persistence parameter:
        else:
            pers = SeasonalPersistence()
            if wndw is None:
                print('Cannot load persistnce seasonal cycle without `wndw` being specified!')
            pers.load(os.path.join(dirs['param_files'],filename_base + 'persistence_seasonal_cycle_{0:d}D-wndw_{1:d}harm.nc'.format(wndw,harm)))
        components_coll['persistence'] = pers
    
    return components_coll