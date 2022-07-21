%load_ext autoreload
%autoreload 2

from pyclarify import ClarifyClient, query, SignalInfo, DataFrame
import matplotlib.pyplot as plt
from forecast_clarify.main import *


client = ClarifyClient('../data/credentials/clarify-credentials_SLS.json')

response = client.select_items(include_dataframe=False)
item_dict = response.result.items
for item_id, meta_data in item_dict.items():
  print(f"ID: {item_id} \t Name: {meta_data.name}")
  
  
item_id = ["ca8dhakpllnee2k58qfg"]

#item_id = ["c983nc7qfsjfsngo9qe0"]

filter = query.Filter(fields={"id": query.In(value=item_id)})
response_unfiltered = client.select_items(
    filter = filter,
    include_metadata = True,
    include_dataframe = True,
    not_before = "2022-07-01T00:00:00Z",
    before = datetime.today()
)
print(response_unfiltered.result.data.times[-1])
# the data (at least for this item) is available with a lag of only two hours


response_filtered = client.select_items(
    filter = filter,
    include_metadata = True,
    include_dataframe = True,
    not_before = "2019-05-01T00:00:00Z",
    before = datetime.today(),
    rollup = "P7DT"
)
# question: is the time tag that comes out of the rollup the center of the window?

st3m_langoey = response_filtered.result.data.to_pandas()
print(st3m_langoey.index[-10:])



response_filtered_1D = client.select_items(
    filter = filter,
    include_metadata = True,
    include_dataframe = True,
    not_before = "2019-05-01T00:00:00Z",
    before = datetime.today(),
    rollup = "P1DT"
)
# question: is the time tag that comes out of the rollup the center of the window?

st3m_langoey_daily = response_filtered_1D.result.data.to_pandas()
print(st3m_langoey_daily.index[-10:])



f,ax = plt.subplots(figsize=(10,4))
st3m_langoey.ca8dhakpllnee2k58qfg_avg.plot(ax=ax)
st3m_langoey.ca8dhakpllnee2k58qfg_min.plot(ax=ax)
st3m_langoey.ca8dhakpllnee2k58qfg_max.plot(ax=ax)
st3m_langoey_daily.ca8dhakpllnee2k58qfg_avg.plot(ax=ax,ls='dotted')
ax.grid()
ax.set_ylabel('{0:s}  {1:s} /{2:s}'.format(
        response_unfiltered.result.items[item_id[0]].name,
        response_unfiltered.result.items[item_id[0]].labels['depth'][0],
        response_unfiltered.result.items[item_id[0]].engUnit
    )
);
ax.set_title('{0:s} {1:s} 7-day average'.format(
        response_unfiltered.result.items[item_id[0]].name,
        response_unfiltered.result.items[item_id[0]].labels['site'][0]
    )
);

#plt.savefig('test.png')
