# pip install pyclarify
from pyclarify import ClarifyClient, query, SignalInfo, DataFrame

client = ClarifyClient("./clarify-credentials.json")

# Get item data and metadata from Clarify
item_id = "ca8dhakpllnee2k58ql0"

filter = query.Filter(fields={"id": query.In(value=[item_id])})
response_1 = client.select_items(filter=filter, include_metadata=True, include_dataframe=True, not_before = "2022-06-01T12:00:00Z", before = "2022-07-10T12:00:00Z")

ts_series = response_1.result.data.series[item_id]
ts_times = response_1.result.data.times

print(ts_series, "\n \n", ts_times)

# Forecast model
values = [1, 2]
dates = ["2021-11-01T21:50:06Z",  "2021-11-02T21:50:06Z"]

# Create a signal and write metadata to it
signal = SignalInfo(name = "My temperature forecast", description = "My Temperature forecast from my data", labels = {"data-source": ["Forecast model"], "location": ["Trondheim"]})
client.save_signals(input_ids=['my_forecast'], signals=[signal], create_only=False)

# Write data into a signal
data = DataFrame(series={"my_forecast": values}, times=dates)
client.insert(data)
