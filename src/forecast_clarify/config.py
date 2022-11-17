import os
import xarray as xr


dirname = os.path.dirname(__file__)

dirs = dict(
    figures=os.path.join(dirname, "/data/processedfigures/"),
    param_files=os.path.join(dirname, "/data/processed/"),
)

bw_sites_file = os.path.join(dirname, "/data/external/barentswatch_sites.json")


def get_datasets():
    datasets = []
    folder = os.path.join(dirname, "data")
    for f in os.listdir(folder):
        sub_folder = os.path.join(folder, f)
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
