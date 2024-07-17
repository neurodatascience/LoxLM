
import pandas as pd
import os
import json
import subprocess
import yaml

"""
Script to scrape DICOM data from json sidecars in openneuro collection.
Uses datalad to download the datasets and scrape the metadata.
Writes to a json file called 'descriptions_start and end index of collection.
"""

bids_suffix_path = "./bids-schema/versions/master/schema/objects/suffixes.yaml"




suffixes = []
with open(bids_suffix_path) as f:
    bids_suffix = yaml.load(f, Loader=yaml.FullLoader)
    for suffix in bids_suffix:
        suffixes.append(suffix)




suffix_endings = [f"{suffix}.json" for suffix in suffixes]


data=pd.read_csv('./src/utils/openneuro.tsv',sep='\t')


names = data['name']
start_index = 0
end_index = len(names)
names_subset = names[start_index:end_index]

def scan_dir(path):
    for root, dirs, files in os.walk(path):
        encodings = ['UTF-8', 'utf-8-sig']
        dic = {}
        files = [file for file in files if file.endswith(tuple(suffix_endings))]
        for file in files:
            d = None
            for encoding in encodings:
                try:
                    with open(f"{root}/{file}", encoding=encoding) as f:
                        d = json.load(f)
                        if "SeriesDescription" in d:
                            sd = d["SeriesDescription"]
                        else:
                            sd = "NA"
                        if "ProtocolName" in d:
                            pn = d["ProtocolName"]
                        else:
                            pn = "NA"
                        tn = d["TaskName"] if "TaskName" in d else "NA"
                        rt = d["RepetitionTime"] if "RepetitionTime" in d else "NA"
                        et = d["EchoTime"] if "EchoTime" in d else "NA"
                        it = d["InversionTime"] if "InversionTime" in d else "NA"
                        pst = d["PulseSequenceType"] if "PulseSequenceType" in d else "NA"
                        fa = d["FlipAngle"] if "FlipAngle" in d else "NA"
                        m = d["Manufacturer"] if "Manufacturer" in d else "NA"
                        mo = d["ManufacturersModelName"] if "ManufacturersModelName" in d else "NA"
                        tn = d["TaskName"] if "TaskName" in d else "NA"
                        dic[file] = { "SeriesDescription" : sd,
                                       "TaskName": tn,
                                       "ProtocolName" : pn,
                                       "RepetitionTime": rt,
                                       "EchoTime": et,
                                       "InversionTime": it,
                                       "PulseSequenceType": pst,
                                       "FlipAngle": fa,
                                       "Manufacturer": m,
                                       "ManufacturersModelName": mo,}
                        break
                except:
                    print(f"Failed to load {root}/{file}")
                
        dirs = [dir for dir in dirs if "." not in dir]
        for dir in dirs:
            little_dic = scan_dir(f"{root}/{dir}")
            dic.update(little_dic)
        return dic
            


dic = {}
for name in names:
    subprocess.run(["datalad", "install","-d","openneuro",f"openneuro/{name}"])
    print(f"installed - {name}")
    dic_temp = scan_dir(f"openneuro/{name}")
    dic.update(dic_temp)
    print(f"scanned - {name}")
with open(f'descriptions_{start_index}-{end_index}.json', 'w') as f:
    json.dump(dic, f)

    


