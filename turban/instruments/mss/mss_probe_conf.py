#!/bin/env python

import sys, os

sys.path.insert(0, os.getcwd().join("atomixpy"))
sys.path.insert(0, os.getcwd())

import json
import atomixpy.atomixrs as mx
import pandas as pd
from io import StringIO


def probeconf_from_header(fname):
    """
    fname: Path to an *.mrd file.
    Yields a template for the probe config json file. Shear probe sensitivity
    must be added afterward.
    """
    hdr = mx.read_mrd_header(fname)

    coef_str = hdr.split("\r\n\r\n")[-1]
    df = pd.read_csv(
        StringIO(coef_str),
        delim_whitespace=True,
        names=[
            "address",
            "instrument",
            "channel",
            "maybe_calibration_type",
            "sensorname",
            "unit",
            "coef1",
            "coef2",
            "coef3",
            "coef4",
            "coef5",
            "coef6",
        ],
    ).dropna()
    config = {}
    config["sensors"] = {}

    for _, row in df.iterrows():
        if row.sensorname != "Dummy":
            config["sensors"][row.sensorname] = {
                "name": row.sensorname,
                "coefficients": [
                    row.coef1,
                    row.coef2,
                    row.coef3,
                    row.coef4,
                    row.coef5,
                    row.coef6,
                ],
                "channel": int(row.channel - 1),
                "calibration_type": row.maybe_calibration_type,  # TODO: DOUBLE CHECK
            }

    return config


def add_shear_probe_sensitivity(cfg, channel_name, sensitivity):
    cfg["sensors"][channel_name]["sensitivity"] = sensitivity
    cfg["sensors"][channel_name]["coefficients"][0] = 1.47133e-6 / sensitivity
    cfg["sensors"][channel_name]["coefficients"][1] = 2.94266e-6 / sensitivity


if __name__ == "__main__":

    # MSS 046
    cfg = probeconf_from_header("/home/doppler/instruments/MSS/CAST0026.MRD")

    # TODO set real values
    add_shear_probe_sensitivity(cfg, "SHEAR_1", 4.40e-4)
    cfg["sensors"]["SHEAR_1"]["serial_number"] = "032"
    cfg["sensors"]["SHEAR_1"]["reference_temperature"] = 21.
    cfg["sensors"]["SHEAR_1"]["calibration_date"] = "2023-09-15"
    add_shear_probe_sensitivity(cfg, "SHEAR_2", 4.86e-4)
    cfg["sensors"]["SHEAR_2"]["serial_number"] = "033"
    cfg["sensors"]["SHEAR_2"]["reference_temperature"] = 21.
    cfg["sensors"]["SHEAR_2"]["calibration_date"] = "2023-09-18"

    with open("probeconf_mss046.json", "w") as f:
        json.dump(cfg, f, indent=4)

    # MSS 053
    cfg = probeconf_from_header("/home/doppler/instruments/MSS/CAST0028.MRD")

    add_shear_probe_sensitivity(cfg, "SHEAR_1", 3.32e-4)
    cfg["sensors"]["SHEAR_1"]["serial_number"] = "116"
    cfg["sensors"]["SHEAR_1"]["reference_temperature"] = 21.
    cfg["sensors"]["SHEAR_1"]["calibration_date"] = "2023-09-15"
    add_shear_probe_sensitivity(cfg, "SHEAR_2", 3.17e-4)
    cfg["sensors"]["SHEAR_2"]["serial_number"] = "149"
    cfg["sensors"]["SHEAR_2"]["reference_temperature"] = 21.
    cfg["sensors"]["SHEAR_2"]["calibration_date"] = ""

    with open("probeconf_mss053.json", "w") as f:
        json.dump(cfg, f, indent=4)
