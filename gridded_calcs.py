import torch
from torch.utils import checkpoint
import numpy
import h5py
import numpy as np
import logging
import sigpy
import sigpy as sp
import cupy
import sigpy.mri as mr
import os

def init_im(ksp_t,dcf_t,coord_t,mps): #ima,deform_adjoint1,ksp,coord,dcf,mps,t,device,tr_per_frame):
  g=0
  for c in range(mps.shape[0]):
    g=g+torch.utils.checkpoint.checkpoint(calculate_sense0b,torch.reshape(ksp_t[c],[-1]),mps[c].cuda(),torch.reshape(coord_t,[-1,3]),torch.reshape(dcf_t,[-1]))
 

  return g
  
  
def calculate_sense0b(ksp,mps_c,coord_t,dcf):
        F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], cupy.array(coord_t), oversamp=1.25, width=4)
        F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
        FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
        resk=(ksp)*dcf
        resk_real=torch.real(torch.squeeze(resk))
        resk_imag=torch.imag(torch.squeeze(resk))
        resk_real=resk_real.unsqueeze(axis=1)
        resk_imag=resk_imag.unsqueeze(axis=1)
        resk_com=torch.cat([resk_real,resk_imag],axis=1)
        g=FH_torch.apply(resk_com)
        g=torch.complex(g[:,:,:,0],g[:,:,:,1])*torch.conj(mps_c)
        ksp=ksp.detach()
     
        return g




