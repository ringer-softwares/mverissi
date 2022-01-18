import os
from typing import Protocol
import tqdm
import itertools
import numpy as np 
from Gaugi import save as gsave
from kepler import load, load_hdf

l_path  = '/home/micael/Documents/NeuralRinger/Jpsiee'

h5_file = os.path.join(l_path, 'data/data17_13TeV.AllPeriods.sgn.probes_lhvloose_EGAM2.bkg.vprobes_vlhvloose_EGAM7.GRL_v97.15bins/' +\
                     'data17_13TeV.AllPeriods.sgn.probes_lhvloose_EGAM2.bkg.vprobes_vlhvloose_EGAM7.GRL_v97.15bins_et{ET}_eta{ETA}.h5')

npz_file = os.path.join(l_path, 'data/data17_13TeV.AllPeriods.sgn.probes_lhvloose_EGAM2.bkg.vprobes_vlhvloose_EGAM7.GRL_v97.15bins/' +\
                     'data17_13TeV.AllPeriods.sgn.probes_lhvloose_EGAM2.bkg.vprobes_vlhvloose_EGAM7.GRL_v97.15bins_et{ET}_eta{ETA}.npz')

t2calo = ['trig_L2_cl_tight', 'trig_L2_cl_medium', 'trig_L2_cl_loose', 'trig_L2_cl_vloose']
for iet, ieta in tqdm.tqdm(itertools.product(range(3), range(5))):
    # open .npz
    h_f = h5_file.format(ET=iet, ETA=ieta)
    n_f = npz_file.format(ET=iet, ETA=ieta)
    r = dict(np.load(n_f, allow_pickle=True))
    # load h5 file
    r['extra_data']     = load_hdf(h_f)[t2calo].values
    r['extra_features'] = t2calo
    print(n_f.split('/')[-1])
    np.savez(n_f.split('/')[-1], **r, alow_pickle=True)#, protocol='savez_compressed')
