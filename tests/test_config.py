from forecast_clarify.src.forecast_clarify.model_registry import Manager
import sys
import unittest
import xarray as xr
import os

model_registry_manager = Manager()
datasets_dir = model_registry_manager.get_dir()


dirs = dict(
    figures=os.path.join(datasets_dir, "processedfigures/"),
    param_files=os.path.join(datasets_dir, "processed/"),
)

sys.path.insert(1, "src/")


class TestConfig(unittest.TestCase):
    def setUp(self):
        print("I am ready")

    def test_dirs(self):
        xr.open_dataset(
            os.path.join(
                dirs["param_files"],
                "temperature_insitu_3m_norkyst800_barentswatch_closest_20060101-20220920_trend.nc",
            )
        )


if __name__ == "__main__":
    unittest.main()
