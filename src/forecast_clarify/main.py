import xarray as xr
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.optimize import leastsq
from scipy.stats import pearsonr


# --------------------SEASONAL CYCLE ESTIMATION--------------------#
class SeasonalCycle:
    """
    seasonal cycle class
    will transform time axis to day of year (thus ignoring hrs, mins, secs!), so no matter what the
    time step is in the data, it will transform them into single day time steps!
    """

    def __init__(
        self, absolute_vals, nharm=3, time_name="time", moment="mean", load_mode=False
    ):
        """
        Initialize a seasonal cycle object.
        Default number of harmonics is 3, which corresponds approximately
        to a low-pass filter with cut-off at 90 days
        """
        self.nharm = nharm
        self.time_name = time_name
        self.moment = moment

        if not load_mode:
            # convert to xarray.Dataset if not already:
            if isinstance(absolute_vals, xr.Dataset):
                self.absolute_vals = absolute_vals
            elif isinstance(absolute_vals, pd.DataFrame):
                self.absolute_vals = pddf2xrds(absolute_vals)
            elif isinstance(absolute_vals, pd.Series):
                self.absolute_vals = pddf2xrds(pd.DataFrame(absolute_vals))
            else:
                print(
                    "<absolute_vals> is of unsupported type {0:}, try converting to xarray.Dataset, pandas.DataFrame or pandas.Series first".format(
                        type(absolute_vals)
                    )
                )

            # group time series by day of year:
            if "month_day" not in self.absolute_vals.coords:
                self.doy = get_doy_coord(self.absolute_vals)
                self.absolute_vals = self.absolute_vals.assign_coords(
                    month_day=self.doy
                )
            else:
                self.doy = self.absolute_vals.month_day

            # compute calendar day mean and std as initial estimate of the seasonal cycle
            if self.moment == "mean":
                self.abs_doy_mean = self.absolute_vals.groupby("month_day").mean(
                    (time_name)
                )
            elif self.moment == "std":
                self.abs_doy_mean = self.absolute_vals.groupby("month_day").std(
                    (time_name)
                )

    def fit(self):
        """
        fit a seasonal cycle to a given time series
        """
        # ,self.offset,self.amplitude,self.phase
        self.mean_sc, self.offset, self.amplitude, self.phase = xr.apply_ufunc(
            seascyc_full,
            self.abs_doy_mean.month_day,
            self.abs_doy_mean,
            self.nharm,
            input_core_dims=[["month_day"], ["month_day"], []],
            output_core_dims=[["month_day"], [], ["hrmc"], ["hrmc"]],
            dask_gufunc_kwargs=dict(output_sizes={"hrmc": self.nharm}),
            vectorize=True,
            dask="parallelized",  # does probably not make a difference atm, could try decorating the seascyc functions with numba
        )

    def predict(self, doy, time_name="time"):

        prediction = xr.apply_ufunc(
            construct_clim,  # functions
            doy,
            self.offset,
            self.amplitude,
            self.phase,  # input
            input_core_dims=[[time_name], [], ["hrmc"], ["hrmc"]],
            output_core_dims=[
                [time_name],
            ],
            vectorize=True,
            dask="parallelized",  # does probably not make a difference atm, could try decorating the seascyc functions with numba
        )

        return prediction

    def training_anomalies(self):
        """
        expand the estimated seasonal cycle to have a value on every time stamp of the input
        """
        # expand seasonal cycle to valid dates:
        self.sc_exp_doy = self.predict(self.doy)
        # compute anomalies:
        if self.moment == "mean":
            self.anomalies = self.absolute_vals - self.sc_exp_doy
        elif self.moment == "std":
            self.anomalies = self.absolute_vals / self.sc_exp_doy

    def load_sc(self, filename):
        """ """

        with xr.open_dataset(filename) as ds:
            self.amplitude = ds.amplitude
            self.phase = ds.phase
            self.offset = ds.offset
            self.nharm = len(ds.phase.hrmc)

    def save(self, filename):
        """ """

        xr.Dataset(
            data_vars=dict(
                amplitude=self.amplitude.temperature,
                phase=self.phase.temperature,
                offset=self.offset.temperature,
            )
        ).to_netcdf(filename)


def seascyc_full(doy, timeseries, harmonics):

    os, a, ph = harm_fit(doy, timeseries, harmonics)
    return construct_clim(doy, os, a, ph), os, a, ph


def harm_fit(doy, timeseries, harmonics):
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
    # treat nans in the timeseries:
    doy = doy[np.isfinite(timeseries.squeeze())]
    timeseries = timeseries[np.isfinite(timeseries)]

    offset = timeseries.mean(axis=0)
    ts_centered = timeseries - offset

    # transform doy to coordinate running from -pi to pi
    x = 2 * np.pi * (doy - 0.5) / 365 - np.pi

    rms = 3.0 * ts_centered.std() / (2**0.5)
    harm_fit_params = []
    for n in range(0, harmonics):

        # function to minimize
        def opt_func(wv):
            return wv[0] * np.sin(x * (n + 1) + wv[1]) - ts_centered
        # opt_func = lambda wv: wv[0] * np.sin(x * (n + 1) + wv[1]) - ts_centered

        # minimize using least squares:
        harm_fit_params.append(leastsq(opt_func, [rms, 0])[0])

    amp, phase = np.array(harm_fit_params).T

    return offset, amp, phase


def construct_clim(doy, offset, amp, phase):
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
    x = 2 * np.pi * (doy - 0.5) / 365 - np.pi

    for n in range(len(amp)):
        timeseries_harmrem += amp[n] * np.sin(x * (n + 1) + phase[n])

    return timeseries_harmrem + offset


# transform pandas dataframe to xarray dataarray:
def pddf2xrds(pddf, varn_ls=[]):

    if not isinstance(varn_ls, list):
        print(
            "<varn_ls> needs to be passed as list even if only one element should be selected!"
        )
        return

    if pddf.index.name is None:
        pddf.index.name = "time"

    if varn_ls:
        xrds = xr.Dataset(pddf[varn_ls])
    else:
        xrds = xr.Dataset(pddf)

    return xrds.assign_coords(time=pddf.index.values)


def month_day_to_doy(moda_str_lst):
    """
    return a 1D np.array of days of year that ignore leap years and just place any Feb 29 AND Mar 1
    on day 60, Mar 2 on day 61 etc. Note that this means that a leap year will get TWO days with doy 61!
    Note that doy for Jan 1 is 1!
    INPUT:
            moda_str_lst:   (list) list of strings in the form "MM-DD" (M: month, D: day)
    OUTPUT:
            np.array of the same length as input that contains the day of year for each
            of the month-day-strings
    """
    doy = []
    dummy_year = 1999

    base_date = datetime(dummy_year - 1, 12, 31)
    for moda_str in moda_str_lst:
        if moda_str == "02-29":
            doy.append(60)
        else:
            doy.append(
                (
                    datetime.strptime(
                        "{0:d}-{1:s}".format(dummy_year, moda_str), "%Y-%m-%d"
                    )
                    - base_date
                ).days
            )

    return np.array(doy)


def get_doy_coord(data, ign_leap=True):
    """ """

    if ign_leap:
        return xr.DataArray(
            month_day_to_doy(data.time.dt.strftime("%m-%d").values),
            coords=dict(time=(["time"], data["time"].values)),
            dims=("time"),
            name="month_day",
        )

    else:
        return xr.DataArray(
            data.time.dt.dayofyear.values,
            coords=dict(time=(["time"], data["time"].values)),
            dims=("time"),
            name="month_day",
        )


class Trend:
    def __init__(self, degree=1):
        """ """
        self.degree = degree

    def fit(
        self, data_array, time_coord="time", origin=np.datetime64("2000-01-01T00:00:00")
    ):

        self.abs_vals = data_array
        self.time_coord = time_coord
        self.origin = origin

        t_int = np.array(self.abs_vals[self.time_coord] - self.origin, dtype=int)
        self.t_int = xr.DataArray(
            t_int,
            dims={self.time_coord: len(self.abs_vals[self.time_coord])},
            coords={self.time_coord: self.abs_vals[self.time_coord]},
        )

        self.polyv = xr.apply_ufunc(
            lin_reg,
            self.t_int,
            self.abs_vals,
            self.degree,
            input_core_dims=[[self.time_coord], [self.time_coord], []],
            output_core_dims=[["pdeg"]],
            dask_gufunc_kwargs=dict(output_sizes={"pdeg": self.degree + 1}),
            vectorize=True,
            dask="parallelized",
        )

        self.polyv = self.polyv.assign_coords(
            pdeg=[ii for ii in range(self.degree, -1, -1)]
        )

        self.polyv.attrs["origin time"] = "{:}".format(self.origin)
        self.polyv.attrs["time units"] = "ns since {:}".format(self.origin)
        self.polyv.attrs["coefficient arrangment"] = "`deg`, `deg-1`, ... `0`"
        self.polyv.attrs["polynomial degree"] = "{0:d}".format(self.degree)

        self.attributes = self.polyv.attrs

        self.mean = self.abs_vals.mean(time_coord)

    def detrend(self):

        trnd_line = xr.apply_ufunc(
            np.polyval,
            self.polyv,
            self.t_int,
            input_core_dims=[["pdeg"], [self.time_coord]],
            output_core_dims=[[self.time_coord]],
            vectorize=True,
            dask="parallelized",
        )

        self.trend_line = trnd_line

        self.data_detrended = self.abs_vals - trnd_line + self.mean

        return self.data_detrended

    def predict(self, prd_time, time_coord="lags"):

        prd_time_int = xr.DataArray(
            np.array(prd_time - self.origin, dtype=int),
            dims=prd_time.dims,
            coords=prd_time.coords,
        )

        trnd_val = xr.apply_ufunc(
            np.polyval,
            self.polyv,
            prd_time_int,
            input_core_dims=[["pdeg"], [time_coord]],
            output_core_dims=[[time_coord]],
            vectorize=True,
            dask="parallelized",
        )

        return trnd_val  # + self.mean

    def load_trend(self, filename):
        """
        load a trend file and skip the self.fit() step
        """

        with xr.load_dataset(filename) as ds:
            self.polyv = ds.polyv
            self.mean = ds.offset

        self.origin = np.datetime64(ds.polyv.attrs["origin time"])
        self.degree = int(ds.polyv.attrs["polynomial degree"])

    def save(self, filename):
        """
        save a trend file so it can be loaded without estimation from the original data
        """

        dataset = xr.Dataset(
            data_vars=dict(polyv=self.polyv.temperature, offset=self.mean.temperature)
        )

        dataset.polyv.attrs = self.attributes

        dataset.to_netcdf(filename)


def lin_reg(prdctr, prdctnd, degree):

    idx = np.isfinite(prdctnd)

    p = np.polyfit(prdctr[idx], prdctnd[idx], degree)

    return p


# --------------------PERSISTENCE FORECAST MODEL--------------------#
class Persistence:
    def __init__(self, lags=4):
        """
        lags in units of timestep of input data
        """
        self.lags = lags

    def fit(self, timeseries):

        self.training_ts = timeseries

        corr = xr.apply_ufunc(
            auto_corr,
            self.training_ts,
            self.lags,
            input_core_dims=[["time"], []],
            output_core_dims=[
                ["lags"],
            ],
            dask_gufunc_kwargs=dict(output_sizes={"lags": self.lags + 1}),
            vectorize=True,
            dask="parallelized",
        )

        self.corr = corr.assign_coords(lags=np.arange(0, 1 + self.lags))

    def predict(self, initial_condition):
        """
        can only handle a single value as initial condition currently
        """

        persistence_fc = self.corr * initial_condition
        # persistence_fc = xr.concat([initial_condition, persistence_fc_lags],dim='time')

        td = [
            persistence_fc.time.values + pd.Timedelta("{0:d}D".format(ddiff * 7))
            for ddiff in persistence_fc.lags.values
        ]
        tdoy = persistence_fc.month_day.values + persistence_fc.lags.values * 7
        self.persistence_fc = persistence_fc.assign_coords(
            dict(time=("lags", td), time_doy=("lags", tdoy))
        ).drop("month_day")

        return self.persistence_fc

    def load(self, filename):
        """ """
        with xr.open_dataset(filename) as ds:
            self.corr = ds.correlation
            self.lags = ds.lags.max().values

    def save(self, filename):
        """ """
        xr.Dataset(data_vars=dict(correlation=self.corr.temperature)).to_netcdf(
            filename
        )


class SeasonalPersistence:
    def __init__(self, lags=4, wndw=91, harmonics=3):
        """
        lags = 4 means lags 1, 2, 3 & 4 weeks
        """
        self.lags = lags + 1
        self.lags_coord = np.arange(1, self.lags)
        self.pers_wndw = wndw
        self.smooth_harm = harmonics

    def fit(self, timeseries, show_progress=True):
        # re-organize time series by doy and pad with doys at beginning and end
        self.timeseries = timeseries.assign_coords(
            month_day=get_doy_coord(timeseries, ign_leap=False)
        )

        timeseries_by_doy = doy_pad(self.timeseries, self.pers_wndw)

        corr = []
        doy_coo = []

        for moda in set(self.timeseries.month_day.values):
            if moda % 10 == 0 and show_progress:
                print("{0:2.1f}% done".format(moda / 365 * 100))
            doy_coo.append(moda)
            i_sel = np.logical_and(
                timeseries_by_doy.month_day >= moda - self.pers_wndw // 2,
                timeseries_by_doy.month_day <= moda + self.pers_wndw // 2,
            )
            corr_lt = []
            for lead_w in self.lags_coord:
                end = timeseries_by_doy.time.max() - pd.Timedelta(
                    "{:}D".format(lead_w * 7)
                )
                i_sel_n = i_sel
                i_sel_n[i_sel.time > end] = False
                init_slice = timeseries_by_doy.isel(time=i_sel_n)

                # add lead days to init_slice.time and check (one by one) if these dates exist
                coll_lag, coll_ini = [], []
                for ins in init_slice:
                    # print(tt.values)
                    lead_date = ins.time + pd.Timedelta("{:}D".format(lead_w * 7))
                    lead_doy = lead_date.month_day + lead_w * 7
                    add_val = timeseries_by_doy.sel(time=lead_date.values)
                    if (add_val.month_day == lead_doy.values).dims:
                        add_val = add_val.sel(time=add_val.month_day == lead_doy.values)

                    # print(lead_doy.values,add_val.month_day.values)
                    if add_val.time.values.size > 0:
                        coll_lag.append(add_val)
                        coll_ini.append(ins)
                    # print(timeseries_by_doy.sel(time = lead_date.values))

                init_slice_ = xr.concat(coll_ini, dim="time")
                lag_slice = xr.concat(coll_lag, dim="time").assign_coords(
                    time=init_slice_.time
                )

                corr_lt.append(xr.corr(init_slice_, lag_slice, dim="time"))
            corr.append(xr.concat(corr_lt, dim=pd.Index(self.lags_coord, name="lags")))

        doy_coo = np.array(doy_coo)
        self.persistence_raw = xr.concat(corr, dim=pd.Index(doy_coo, name="month_day"))

        # do smoothing with `self.smooth_harm` harmonics if specified (skipped if None)
        if self.smooth_harm is not None:
            pers_type = "smoothed"
            SC_pers = xr.apply_ufunc(
                seascyc_full,
                self.persistence_raw.month_day,
                self.persistence_raw,
                self.smooth_harm,
                input_core_dims=[["month_day"], ["month_day"], []],
                output_core_dims=[["month_day"], [], ["hrmc"], ["hrmc"]],
                dask_gufunc_kwargs=dict(output_sizes={"hrmc": self.smooth_harm}),
                vectorize=True,
                dask="parallelized",
            )

            self.persistence_smooth = xr.apply_ufunc(
                construct_clim,  # functions
                self.persistence_raw.month_day.values,
                SC_pers[1],
                SC_pers[2],
                SC_pers[3],  # input
                input_core_dims=[["month_day"], [], ["hrmc"], ["hrmc"]],
                output_core_dims=[
                    ["month_day"],
                ],
                vectorize=True,
                dask="parallelized",
            ).assign_coords(month_day=doy_coo)
        else:
            pers_type = "raw"
            self.smooth_harm = -1
            self.persistence_smooth = self.persistence_raw

        self.persistence = xr.Dataset(
            data_vars=dict(correlation=self.persistence_smooth),
            attrs=dict(
                persistence_type=pers_type,
                harmonics=self.smooth_harm,
                window=self.pers_wndw,
            ),
        )

    def predict(self, initial_condition):
        self.initial_condition = initial_condition.expand_dims(
            dim={"month_day": [initial_condition.month_day.values]}
        )

        # pad the correlation array with 1s to also return the initial value (lag 0)

        self.corr_pad = self.corr.pad(
            {"lags": (1, 0)}, mode="constant", constant_values=1
        ).assign_coords(lags=np.concatenate([[0], self.corr.lags.values]))
        persistence_fc = self.initial_condition * self.corr_pad

        # get doy of initial value and
        td = [
            persistence_fc.time.values + pd.Timedelta("{0:d}D".format(ddiff * 7))
            for ddiff in persistence_fc.lags.values
        ]
        tdoy = persistence_fc.month_day.values + persistence_fc.lags.values * 7
        self.persistence_fc = (
            persistence_fc.assign_coords(
                dict(time=("lags", td), time_doy=("lags", tdoy))
            )
            .squeeze()
            .drop("month_day")
        )

        return self.persistence_fc

    def load(self, filename):
        with xr.open_dataset(filename) as ds:
            self.corr = ds.correlation
            self.lags = ds.lags.max().values.item()
            self.smooth_harm = ds.attrs["harmonics"]
            self.pers_wndw = ds.attrs["window"]

    def save(self, filename):
        """
        currently saves the a value for every station and doy (365) but could in theory save
        smoothed correlations by just storing `self.smooth_harm`*2 + 1 values per stations
        """
        self.persistence.to_netcdf(filename)


def auto_corr(timeseries, lags):
    timeseries = timeseries[np.isfinite(timeseries)]
    return np.concatenate(
        [
            np.array([1]),
            np.array(
                [
                    pearsonr(timeseries[lg:], timeseries[:-lg])[0]
                    for lg in range(1, 1 + lags)
                ]
            ),
        ]
    )


def doy_pad(timeseries, wndw):
    st_y = timeseries.time.dt.year.min().values.item()
    en_y = timeseries.time.dt.year.max().values.item()
    pad = int(np.ceil((en_y - st_y) / 7)) * wndw // 2

    timeseries_reorg = xr.DataArray(
        timeseries.temperature.values,
        dims={
            "month_day": len(timeseries.time.values),
            "location": len(timeseries.location.values),
        },
        coords={
            "location": ("location", timeseries.location.values),
            "month_day": ("month_day", timeseries.month_day.values),
            "time": ("month_day", timeseries.time.values),
        },
    ).sortby("month_day")
    timeseries_reorg = timeseries_reorg.pad(month_day=pad, mode="wrap")
    new_month_day = timeseries_reorg.month_day.values
    new_month_day[:pad] -= 365
    new_month_day[-pad:] += 365

    return xr.DataArray(
        timeseries_reorg.values,
        dims={
            "time": len(timeseries_reorg.time.values),
            "location": len(timeseries_reorg.location.values),
        },
        coords={
            "location": ("location", timeseries_reorg.location.values),
            "month_day": ("time", new_month_day),
            "time": ("time", timeseries_reorg.time.values),
        },
    )
