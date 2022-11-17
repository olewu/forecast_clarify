import sys
import unittest
import xarray as xr
import os


sys.path.insert(1, "src/")
from forecast_clarify.config import dirs


class TestConfig(unittest.TestCase):
    def setUp(self):
        print("I am ready")

    def test_dirs(self):
        ds = xr.open_dataset(
            os.path.join(
                dirs["param_files"],
                "temperature_insitu_3m_norkyst800_barentswatch_closest_20060101-20220920_trend.nc",
            )
        )


if __name__ == "__main__":
    unittest.main()
