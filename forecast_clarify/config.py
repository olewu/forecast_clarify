import os

base = '/projects/NS9001K/owul/projects/forecast_clarify/'

dirs = dict(
    proj_base   = base,
    figures     = os.path.join(base,'data/processedfigures/'),
    param_files = os.path.join(base,'data/processed/')
)

bw_sites_file = os.path.join(dirs['proj_base'],'data/external/barentswatch_sites.json')