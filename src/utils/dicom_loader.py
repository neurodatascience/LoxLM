from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pydicom


class DicomLoader:
    """Process dicom header information for model ingestion."""

    def __init__(
        self,
        path,
    ):
        dicoms = self.scan_dicom_dir(path)
        clean_df = self.clean_dicoms(dicoms)
        self.clean_dict = clean_df.to_dict(orient="records")

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
                    rt = float(d.RepetitionTime) if "RepetitionTime" in d else np.nan
                    et = float(d.EchoTime) if "EchoTime" in d else np.nan
                    it = float(d.InversionTime) if "InversionTime" in d else np.nan
                    pst = d.PulseSequenceType if "PulseSequenceType" in d else "NA"
                    fa = float(d.FlipAngle) if "FlipAngle" in d else np.nan
                    m = d.Manufacturer if "Manufacturer" in d else "NA"
                    mo = d.ManufacturersModelName if "ManufacturersModelName" in d else "NA"
                    tn = d.TaskName if "TaskName" in d else "NA"
                    dic = {
                        "file": file,
                        "root": root,
                        "series_description": sd,
                        "task_name": tn,
                        "protocol_name": pn,
                        "repetition_time": rt,
                        "echo_time": et,
                        "inversion_time": it,
                        "pulse_sequence_type": pst,
                        "flip_angle": fa,
                        "manufacturer": m,
                        "model": mo,
                    }
                    dicoms.append(dic)
                except Exception as e:
                    print(f"{e}\nFailed to load {root}/{file}")
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

    def get_dict(self):
        return self.clean_dict
