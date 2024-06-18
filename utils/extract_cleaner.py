from __future__ import annotations

import json
import os

import pandas as pd

dir = "../openneuro_extracts"

data = {}
for root, dirs, files in os.walk(dir):
    for file in files:
        with open(f"{root}/{file}") as f:
            data.update(json.load(f))

df = pd.DataFrame.from_dict(data, orient="index")

useful = df[(df["SeriesDescription"] != "NA") | (df["ProtocolName"] != "NA")]

useful.reset_index(inplace=True)

# Rename filenames by removing everything before the last underscore and create new 'name' column
useful["index"] = useful["index"].apply(lambda x: x.rsplit("_", 1)[-1])
useful["index"] = useful["index"].apply(lambda x: x.split(".")[0])

useful.drop_duplicates(subset=["index", "SeriesDescription", "ProtocolName"], inplace=True)

with open("examples_all.json", "w") as f:
    useful.to_json(
        f,
        orient="records",
    )
