import os

dirs = dict(
    figures     = os.path.relpath('src/data/processedfigures/'),
    param_files = os.path.relpath('src/data/processed/')
)

bw_sites_file = os.path.relpath('src/data/external/barentswatch_sites.json')