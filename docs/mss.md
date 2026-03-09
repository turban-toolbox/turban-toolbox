# Turban Mikrostruktur Sonde (MSS) implementation reference

The MSS is a shear microstructure profiles produced by Sea & Sun Technology (SST). 

See here for a documentation of the development history (link to Matthäus article, to be added soon).

## Technical setup of a MSS

A standard MSS is connected by a 4-wire cable that provides power (2-wires) and a uni-directional
2-wire RS422 connection with 600 kbit [^1].
If the device is powered, it will send raw binary data in the HHL-Format using the RS422 interface.
In a typical setup the MSS is connected to a computer with the SST-SDA software, which is capable to collect
raw MSS data together with a GPS device. SST-SDA writes the data in the MRD format, combining
MSS and optional GPS data in binary format but with a verbose header of the measurement setup.
The typical workflow is to create a configuration for a specific MSS and to process a bunch of MRD files.
This will be explained below. For processing the HHL data refer to HHL API reference below. 


The configuration of an MSS is done via the [`MssDeviceConfig`][turban.instruments.mss.config].

[^1]: Check baud rate!

## MSS API-Reference

[MSS-Config](instruments/mss/config.md)

[MRD (MSS Raw Data)](instruments/mss/mss_mrd.md)

[HHL (High-High-Low)](instruments/mss/mss_hhl.md)






