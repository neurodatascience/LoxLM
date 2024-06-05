# %%
import pandas as pd
import os
import json
import subprocess
import yaml

bids_suffix_path = "./bids-schema/versions/master/schema/objects/suffixes.yaml"



# %%
suffixes = []
with open(bids_suffix_path) as f:
    bids_suffix = yaml.load(f, Loader=yaml.FullLoader)
    for suffix in bids_suffix:
        suffixes.append(suffix)



# %%
suffix_endings = [f"{suffix}.json" for suffix in suffixes]

# %%

# %%
data=pd.read_csv('./utils/openneuro.tsv',sep='\t')

# %%
names = data['name']
start_index = 200
end_index = 250
names_subset = names[start_index:end_index]

# %%

# %%
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
                        dic[file] = { "SeriesDescription" : sd,  "ProtocolName" : pn}
                        break
                except:
                    print(f"Failed to load {root}/{file}")
                
        dirs = [dir for dir in dirs if "." not in dir]
        for dir in dirs:
            little_dic = scan_dir(f"{root}/{dir}")
            dic.update(little_dic)
        return dic
            

# %%
dic = {}
for name in names_subset:
    subprocess.run(["datalad", "install","-d","openneuro",f"openneuro/{name}"])
    print(f"installed - {name}")
    dic_temp = scan_dir(f"openneuro/{name}")
    dic.update(dic_temp)
    print(f"scanned - {name}")
with open(f'descriptions_{start_index}-{end_index}', 'w') as f:
    json.dump(dic, f)

    


