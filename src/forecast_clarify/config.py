import os

import xarray as xr
from forecast_clarify.model_registry import Manager

model_registry_manager = Manager()

def get_datasets():
    datasets_dir = model_registry_manager.get_dir()
    datasets = []
    for f in os.listdir(datasets_dir):
        sub_folder = os.path.join(datasets_dir, f)
        if ".py" not in sub_folder and "__" not in sub_folder:
            for d in os.listdir(sub_folder):
                if d.split(".")[-1] in ["json", "nc"]:
                    dataset = os.path.join(sub_folder, d)
                    datasets.append(dataset)
    return datasets


def list_dataset():
    datasets = get_datasets()
    for d in datasets:
        print(d)


def load_dataset(filename):
    return xr.open_dataset(filename)
