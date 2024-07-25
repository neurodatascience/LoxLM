from __future__ import annotations

import os

import pandas as pd
import pydicom


class Dicom_Loader:
    """Process dicom header information for model ingestion."""

    def __init__(
        self,
        path,
    ):
        dicoms = self.scan_dicom_dir(path)
        clean_df = self.clean_dicoms(dicoms)
        return clean_df.to_dict(orient="records")

    def scan_dicom_dir(self, path):
        for root, dirs, files in os.walk(path):
            dicoms = []
            for file in files:
                if not file.endswith(".dcm"):
                    continue
                try:
                    d = pydicom.dcmread(f"{root}/{file}")
                    if "SeriesDescription" in d:
                        sd = d.SeriesDescription
                    else:
                        sd = "NA"
                    if "ProtocolName" in d:
                        pn = d.ProtocolName
                    else:
                        pn = "NA"
                    tn = d.TaskName if "TaskName" in d else "NA"
                    rt = d.RepetitionTime if "RepetitionTime" in d else "NA"
                    et = d.EchoTime if "EchoTime" in d else "NA"
                    it = d.InversionTime if "InversionTime" in d else "NA"
                    pst = d.PulseSequenceType if "PulseSequenceType" in d else "NA"
                    fa = d.FlipAngle if "FlipAngle" in d else "NA"
                    m = d.Manufacturer if "Manufacturer" in d else "NA"
                    mo = d.ManufacturersModelName if "ManufacturersModelName" in d else "NA"
                    tn = d.TaskName if "TaskName" in d else "NA"
                    dic = {
                        "file": file,
                        "root": root,
                        "SeriesDescription": sd,
                        "TaskName": tn,
                        "ProtocolName": pn,
                        "RepetitionTime": rt,
                        "EchoTime": et,
                        "InversionTime": it,
                        "PulseSequenceType": pst,
                        "FlipAngle": fa,
                        "Manufacturer": m,
                        "ManufacturersModelName": mo,
                    }
                    dicoms.append(dic)
                except Exception:
                    print(f"Failed to load {root}/{file}")
            dirs = [dir for dir in dirs if "." not in dir]
            for dir in dirs:
                little_dic = self.scan_dicom_dir(f"{root}/{dir}")
                dicoms = dicoms + little_dic
            return dicoms

    def clean_dicoms(self, dicoms):
        df = pd.DataFrame(dicoms)

        sub = df.columns.to_list()
        sub.remove("file")
        unique_rows = df.drop_duplicates(subset=sub)
        return unique_rows
