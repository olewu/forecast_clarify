# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The changelog for SDK version 0.x.x can be found here.

Changes are grouped as follows

- `Added` for new features.
- `Changed` for changes in existing functionality.
- `Deprecated` for soon-to-be removed features.
- `Removed` for now removed features.
- `Fixed` for any bug fixes.
- `Security` in case of vulnerabilities.

## [0.1.0] - 2022-11-10

### Removed

- `setup.py`
- `requirements_minimal.txt`
- `environment.yml`
- `.pre-commit-config.yaml`

### Changed

- Moved `forecast_clarify` to `src` folder
- Moved `data` folder to `forecast_clarify` folder.
- Changed name of `main.seas_cycle` class to `main.SeasonalCycle`
- Changed name of `main.trend` class to `main.Trend`
- Changed name of `main.persistence` class to `main.Persistence`
- Changed name of `main.persistence_seasonal` class to `main.SeasonalPersistence`
- Removed unused requirements in `requirements.txt`
- Changed hardcoded path to files in `forecast_clarify.data` to relative path

### Added

- `CHANGELOG.md` file
- `load_dataset`, `list_datasets` and `get_datasets` methods.

## [0.1.1] - 2022-11-17

### Changed

- conform with PEP8-standard
- relaxed xarray version requirement in `setup.cfg`
- Changed hardcoded index for dataset in `forecast_clarfiy.clarify_persistece_package.find_station_in_bw` to look-up of `.json` file
- added changes to `CHANGELOG.md`
- update `README.md`


## 2022-11-29

### Removed

- `src/forecast_clarify/data/__init__.py`
- `src/forecast_clarify/data/external/.gitkeep`
- `src/forecast_clarify/data/external/__init__.py`
- `src/forecast_clarify/data/external/barentswatch_sites.json`
- `src/forecast_clarify/data/processed/.gitkeep`
- `src/forecast_clarify/data/processed/__init__.py`
- `src/forecast_clarify/data/processed/temperature_insitu_3m_norkyst800_barentswatch_closest_20060101-20220920_persistence.nc`
- `src/forecast_clarify/data/processed/temperature_insitu_3m_norkyst800_barentswatch_closest_20060101-20220920_persistence_seasonal_cycle_61D-wndw_3harm.nc`
- `src/forecast_clarify/data/processed/temperature_insitu_3m_norkyst800_barentswatch_closest_20060101-20220920_seasonal_cycle.nc`
- `src/forecast_clarify/data/processed/temperature_insitu_3m_norkyst800_barentswatch_closest_20060101-20220920_seasonal_cycle_std.nc`
- `src/forecast_clarify/data/processed/temperature_insitu_3m_norkyst800_barentswatch_closest_20060101-20220920_trend.nc`
- `src/forecast_clarify/data/raw/.gitkeep`
- `src/forecast_clarify/data/raw/__init__.py`

### Added

- Implemented model registry manager, which can pull/push datasets from W&B: `src/forecast_clarify/model_registry.py`

### Changed

- All files from `data/` are moved into <a href="https://wandb.ai/clarify/climate-futures-clarify/artifacts/dataset/sea_temperature/v0">About<a>, from which the model registry manager pulls the model parameters
- The only requirement is to set the WANDB_TOKEN env variable before execution.
- added wandb as requirement in `requirements.txt`
- added test of model registry manager to `tests/test_config.py`