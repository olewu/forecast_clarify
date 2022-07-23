import xarray as xr
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.optimize import leastsq

#--------------------SEASONAL CYCLE ESTIMATION--------------------#
class seas_cycle():
    """
    seasonal cycle class
    will transform time axis to day of year (thus ignoring hrs, mins, secs!), so no matter what the
    time step is in the data, it will transform them into single day time steps!
    """
    
    def __init__(self,absolute_vals,nharm=3,time_name='time'):
        """
        Initialize a seasonal cycle object.
        Default number of harmonics is 3, which corresponds approximately
        to a low-pass filter with cut-off at 90 days
        """
        self.nharm = nharm
        self.time_name = time_name

        # convert to xarray.Dataset if not already:
        if isinstance(absolute_vals,xr.Dataset):
            self.absolute_vals = absolute_vals
        elif isinstance(absolute_vals,pd.DataFrame):
            self.absolute_vals = pddf2xrds(absolute_vals)
        elif isinstance(absolute_vals,pd.Series):
            self.absolute_vals = pddf2xrds(pd.DataFrame(absolute_vals))
        else:
            print('<absolute_vals> is of unsupported type {0:}, try converting to xarray.Dataset, pandas.DataFrame or pandas.Series first'.format(type(absolute_vals)))

        # group time series by day of year:
        if 'month_day' not in self.absolute_vals.coords:
            self.doy = get_doy_coord(self.absolute_vals)
            self.absolute_vals = self.absolute_vals.assign_coords(month_day=self.doy)
        
        # compute calendar day mean and std as initial estimate of the seasonal cycle
        self.abs_doy_mean = self.absolute_vals.groupby('month_day').mean((time_name))

    def fit(self):
        """
        fit a seasonal cycle to a given time series
        """
        # ,self.offset,self.amplitude,self.phase
        self.mean_sc,self.offset,self.amplitude,self.phase = xr.apply_ufunc(
            seascyc_full,
            self.abs_doy_mean.month_day,
            self.abs_doy_mean,
            self.nharm,
            input_core_dims = [['month_day'],['month_day'],[]],
            output_core_dims = [['month_day'],[],['hrmc'],['hrmc']],
            dask_gufunc_kwargs =  dict(output_sizes = {'hrmc':self.nharm}),
            vectorize = True,
            dask = 'parallelized' # does probably not make a difference atm, could try decorating the seascyc functions with numba
        )

    def predict(self,doy,time_name = 'time'):

        prediction = xr.apply_ufunc(
            construct_clim, # functions
            doy, self.offset,self.amplitude,self.phase, # input
            input_core_dims = [[time_name],[],['hrmc'],['hrmc']],
            output_core_dims = [[time_name],],
            vectorize = True,
            dask = 'parallelized' # does probably not make a difference atm, could try decorating the seascyc functions with numba
        )

        return prediction

    def training_anomalies(self):
        """
        expand the estimated seasonal cycle to have a value on every time stamp of the input
        """
        # expand seasonal cycle to valid dates:
        self.sc_exp_doy = self.predict(self.doy)
        # compute anomalies:
        self.anomalies = self.absolute_vals - self.sc_exp_doy

def seascyc_full(doy,timeseries,harmonics):

    os,a,ph = harm_fit(doy,timeseries,harmonics)
    return construct_clim(doy,os,a,ph),os,a,ph


def harm_fit(doy,timeseries,harmonics):
    """
    obtain a given number of harmonics from a time series
    
    INPUT:
            doy:            (1D np.array) 
            timeseries:     (1D np.array) corresponding time series  
            harmonics:      (list)
    OUTPUT:
            returns offest, amp, phase (int,list,list) where amp and phase have length
            matching the number of requested harmonics 
    """

    offset = timeseries.mean(axis=0)
    ts_centered = timeseries - offset

    # transform doy to coordinate running from -pi to pi
    x = 2*np.pi*(doy - .5)/365 - np.pi
    
    rms = 3.*ts_centered.std()/(2**.5)
    harm_fit_params = []
    for n in range(0,harmonics):
    
        # function to minimize
        opt_func = lambda wv: wv[0] * np.sin(x*(n+1) + wv[1]) - ts_centered
    
        # minimize using least squares:
        harm_fit_params.append(leastsq(opt_func,[rms,0])[0])
    
    amp,phase = np.array(harm_fit_params).T
    
    return offset,amp,phase


def construct_clim(doy,offset,amp,phase):
    """
    construct the climatology from a given number of harmonics
    
    INPUT:
            doy:        (list) 
            offest:     (float) 
            amp:        (list) 
            phase:      (list) 
    OUTPUT:
            np.array of length 'ts_len' 
    """

    timeseries_harmrem = np.zeros([len(doy)])
    # transform doy to coordinate running from -pi to pi
    x = 2*np.pi*(doy - .5)/365 - np.pi

    for n in range(len(amp)):
        timeseries_harmrem += amp[n]*np.sin(x*(n+1) + phase[n])

    return timeseries_harmrem + offset


# transform pandas dataframe to xarray dataarray:
def pddf2xrds(pddf,varn_ls=[]):
    
    if not isinstance(varn_ls,list):
        print('<varn_ls> needs to be passed as list even if only one element should be selected!')
        return

    if pddf.index.name is None:
        pddf.index.name = 'time'

    if varn_ls:
        xrds = xr.Dataset(
            pddf[varn_ls]
        )
    else:
        xrds = xr.Dataset(
            pddf
        )

    return xrds.assign_coords(time=pddf.index.values)

def month_day_to_doy(moda_str_lst):
    """
    return a 1D np.array of days of year that ignore leap years and just place any Feb 29 AND Mar 1
    on day 61, Mar 2 on day 62 etc. Note that this means that a leap year will get TWO days with doy 61!
    Note that doy for Jan 1 is 1!
    INPUT:
            moda_str_lst:   (list) list of strings in the form "MM-DD" (M: month, D: day)
    OUTPUT:
            np.array of the same length as input that contains the day of year for each 
            of the month-day-strings
    """
    doy = []
    dummy_year = 1999
    base_date = datetime(dummy_year-1,12,31)
    for moda_str in moda_str_lst:
        if moda_str == '02-29':
            doy.append(60)
        else:
            doy.append(
                (datetime.strptime('{0:d}-{1:s}'.format(dummy_year,moda_str),'%Y-%m-%d') - base_date).days
            )

    return np.array(doy)

def get_doy_coord(data):
    """
    """

    return xr.DataArray(
        month_day_to_doy(data.time.dt.strftime('%m-%d').values),
        coords=dict(time=(['time'],data['time'].values)),
        dims=('time'),
        name='month_day'
        )


#--------------------PERSISTENCE FORECAST MODEL--------------------#
class persistence():
    def __init__(self,lags=4):
        """
        lags in units of timestep of input data
        """
        self.lags = lags

    def fit(self,timeseries):
        
        self.training_ts = timeseries
        
        corr = xr.apply_ufunc(
            auto_corr,
            self.training_ts,self.lags,
            input_core_dims = [['time'],[]],
            output_core_dims = [['lags'],],
            dask_gufunc_kwargs =  dict(output_sizes = {'lags':self.lags+1}),
            vectorize = True,
            dask = 'parallelized'
        )

        self.corr = corr.assign_coords(lags=np.arange(0,1+self.lags))

    def predict(self,initial_condition):
        """
        can only handle a single value as initial condition currently
        """
        
        persistence_fc = self.corr * initial_condition
        # persistence_fc = xr.concat([initial_condition, persistence_fc_lags],dim='time')
        
        td = [persistence_fc.time.values + pd.Timedelta('{0:d}D'.format(ddiff*7)) for ddiff in persistence_fc.lags.values]
        tdoy = persistence_fc.month_day.values + persistence_fc.lags.values*7
        self.persistence_fc = persistence_fc.assign_coords(dict(time = ('lags',td),time_doy = ('lags',tdoy))).drop('month_day')

        return self.persistence_fc

def auto_corr(timeseries,lags):
    return np.concatenate([np.array([1]),np.array([np.corrcoef(timeseries[lg:],timeseries[:-lg])[0,1] for lg in range(1,1+lags)])])