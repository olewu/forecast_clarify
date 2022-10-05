import os

base = '/projects/NS9001K/owul/'

dirs = dict(
    proj_base   = os.path.join(base,'projects/forecast_clarify/'),
    figures     = os.path.join(base,'figures/S2S_fishfarm/'),
)
dirs['param_files'] = os.path.join(dirs['proj_base'],'data/processed/')

bw_sites_file = os.path.join(dirs['proj_base'],'data/external/barentswatch_sites.json')