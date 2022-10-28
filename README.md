forecast_clarify
==============================
[![Build Status](https://github.com/olewu/forecast_clarify/workflows/Tests/badge.svg)](https://github.com/olewu/forecast_clarify/actions)
[![codecov](https://codecov.io/gh/olewu/forecast_clarify/branch/main/graph/badge.svg)](https://codecov.io/gh/olewu/forecast_clarify)
[![License:MIT](https://img.shields.io/badge/License-MIT-lightgray.svg?style=flt-square)](https://opensource.org/licenses/MIT)[![pypi](https://img.shields.io/pypi/v/forecast_clarify.svg)](https://pypi.org/project/forecast_clarify)
<!-- [![conda-forge](https://img.shields.io/conda/dn/conda-forge/forecast_clarify?label=conda-forge)](https://anaconda.org/conda-forge/forecast_clarify) -->[![Documentation Status](https://readthedocs.org/projects/forecast_clarify/badge/?version=latest)](https://forecast_clarify.readthedocs.io/en/latest/?badge=latest)


Local water temperature (3m) weekly forecasts based on local statistics (seasonal cycle, week-to-week persistence) from NorKyst800 (2006 - 2022).

To install, initialize an environment with python (tested with 3.10.0 and 3.10.4 but expect other version of python3 to work).

Install the minimal requirements using
```
pip install -r requirements_minimal.txt
```
then install the package functionality from the project root folder (where this README is located) using
```
pip install -e .
```

This should enable you to run `/notebooks/010_test_load.ipynb`. The notebook will access functionality in `/forecast_clarify/clarify_persistence_package.py`, which in turn needs `/forecast_clarify/main.py` and `/forecast_clarify/config.py` as well as the model parameter files saved in `/data/processed/`.


--------

<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-project">cookiecutter science project template</a>.</small></p>
