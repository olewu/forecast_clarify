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
