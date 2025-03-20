import turban


mrdfile = 'Nien0020.MRD'
mrddata = turban.instruments.mss_mrd.mrd(mrdfile)
print(mrddata.level0['PRESS'])
