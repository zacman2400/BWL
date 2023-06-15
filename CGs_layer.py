#use for unrolls

import h5py
import numpy as np
import logging
import sigpy
import sigpy as sp
import cupy
import sigpy.mri as mr
import os
import torch
from torch.utils import checkpoint
def initialize0(M_ta,ksp,mps_c,coord_t,dcf):
 
    F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], coord_t.detach().cpu().numpy(), oversamp=1.25, width=4)
    F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
    FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
      
 
    k=(ksp)*dcf
    k_real=torch.real(torch.squeeze(k))
    k_imag=torch.imag(torch.squeeze(k))
    k_real=k_real.unsqueeze(axis=2)
    k_imag=k_imag.unsqueeze(axis=2)
    k_com=torch.cat([k_real,k_imag],axis=2)
    
    g=FH_torch.apply(k_com)
    y_guess=torch.complex(g[:,:,:,0],g[:,:,:,1])*torch.conj(mps_c).cuda()
   
    Ap0=torch.utils.checkpoint.checkpoint(operator0,M_ta,mps_c,dcf,coord_t)
    resk=y_guess-Ap0
    p=resk
    rzold=torch.real(torch.vdot(resk.flatten(),resk.flatten()))
    rzold=rzold
    p=p
    resk=resk
    torch.cuda.empty_cache()
    return rzold,p,resk
    
    
def update_CG0(M_ta,ksp,mps,coord_t,dcf,rzold,p,resk):
    F = sp.linop.NUFFT([mps.shape[1], mps.shape[2], mps.shape[3]], coord_t.detach().cpu().numpy(), oversamp=1.25, width=4)
    F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
    FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
      
  
    Ap=0

    for c in range(mps.shape[0]):
        Ap=Ap+torch.utils.checkpoint.checkpoint(operator,p,mps[c],dcf,coord_t)
  
    
    pAp=torch.real(torch.vdot(p.flatten(),Ap.flatten()))
   
    alpha=rzold/pAp
 
    alpha=alpha.cuda()
   
    M_ta=M_ta+alpha*p
    resk=resk-alpha*Ap
    
    rznew=torch.real(torch.vdot(resk.flatten(),resk.flatten()))
    beta=rznew/rzold
    
 
    beta=beta.cuda()
    
    p=resk+beta*p
   
    torch.cuda.empty_cache()
    return rznew,p,M_ta,resk
  
def operator0(x,mps_c,dcf,coord_t):
 
    F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], coord_t.detach().cpu().numpy(), oversamp=1.25, width=4)
    F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
    FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
    mps_c=mps_c.cuda()
    diff=torch.cat([torch.reshape(torch.real(x*mps_c),[mps_c.shape[0],mps_c.shape[1],mps_c.shape[2],1]),torch.reshape(torch.imag(x*mps_c),[mps_c.shape[0],mps_c.shape[1],mps_c.shape[2],1])],axis=3)
    
    e_tc=F_torch.apply(diff)
  
    e_tca=torch.complex(e_tc[:,:,0],e_tc[:,:,1])*dcf
    e_tca_real=torch.real(torch.squeeze(e_tca))
    e_tca_imag=torch.imag(torch.squeeze(e_tca))
    e_tca_real=e_tca_real.unsqueeze(axis=2)
    e_tca_imag=e_tca_imag.unsqueeze(axis=2)
    e_tca_com=torch.cat([e_tca_real,e_tca_imag],axis=2)
    g=FH_torch.apply(e_tca_com)

    gout=torch.complex(g[:,:,:,0],g[:,:,:,1])*torch.conj(mps_c).cuda() #+.0001*x
 
   
    return gout 
    
def initialize(M_ta,ksp,mps_c,coord_t,dcf,alpha):
   # print(mps_c.shape)
    F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], coord_t.detach().cpu().numpy(), oversamp=1.25, width=2)
    F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
    FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
      

    k=(ksp)*dcf
   
    k_real=torch.real(torch.squeeze(k))
    k_imag=torch.imag(torch.squeeze(k))
    k_real=k_real.unsqueeze(axis=2)
    k_imag=k_imag.unsqueeze(axis=2)
    k_com=torch.cat([k_real,k_imag],axis=2)
    
    g=FH_torch.apply(k_com)
    y_guess=torch.complex(g[:,:,:,0],g[:,:,:,1])*torch.conj(mps_c).cuda()
   
    Ap0=torch.utils.checkpoint.checkpoint(operator0,M_ta,mps_c,dcf,coord_t)
 
    resk=y_guess-Ap0
    p=resk
    rzold=torch.real(torch.vdot(resk.flatten(),resk.flatten()))
    rzold=rzold
    p=p
    resk=resk
    torch.cuda.empty_cache()
    return rzold,p,resk
    
    
def update_CG(M_ta,ksp,mps,coord_t,dcf,rzold,p,resk,alpha1):
    F = sp.linop.NUFFT([mps.shape[1], mps.shape[2], mps.shape[3]], coord_t.detach().cpu().numpy(), oversamp=1.25, width=4)
    F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
    FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
      
    
  
    Ap=0

    for c in range(mps.shape[0]):
       
        Ap=Ap+torch.utils.checkpoint.checkpoint(operator,p,mps[c],dcf,coord_t,alpha1)
    Ap=Ap.cuda()
    pAp=torch.real(torch.vdot(p.flatten(),Ap.flatten()))
   
    alpha=rzold/pAp
   # alpha=np.float32(alpha)*torch.ones([1])
    alpha=alpha.cuda()
   
    M_ta=M_ta+alpha*p
    resk=resk-alpha*Ap
    
    rznew=torch.real(torch.vdot(resk.flatten(),resk.flatten()))
    beta=rznew/rzold
    
 
    beta=beta.cuda()
    
    p=resk+beta*p
   
    torch.cuda.empty_cache()
    return rznew,p,M_ta,resk
  
def operator(x,mps_c,dcf,coord_t,alpha):
    F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], coord_t.detach().cpu().numpy(), oversamp=1.25, width=4)
    F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
    FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
    mps_c=mps_c.cuda()
    diff=torch.cat([torch.reshape(torch.real(x*mps_c),[mps_c.shape[0],mps_c.shape[1],mps_c.shape[2],1]),torch.reshape(torch.imag(x*mps_c),[mps_c.shape[0],mps_c.shape[1],mps_c.shape[2],1])],axis=3)
    
    e_tc=F_torch.apply(diff)
 
    e_tca=torch.complex(e_tc[:,:,0],e_tc[:,:,1])*dcf
    e_tca_real=torch.real(torch.squeeze(e_tca))
    e_tca_imag=torch.imag(torch.squeeze(e_tca))
    e_tca_real=e_tca_real.unsqueeze(axis=2)
    e_tca_imag=e_tca_imag.unsqueeze(axis=2)
    e_tca_com=torch.cat([e_tca_real,e_tca_imag],axis=2)
    g=FH_torch.apply(e_tca_com)
 
    gout=torch.complex(g[:,:,:,0],g[:,:,:,1]).cuda()*torch.conj(mps_c).cuda()+alpha.cuda()*x.cuda()

  
    return gout
def CG_DClayer(out0,ksp,coord,dcf,mps,alpha,iters):
    for n in range(1):
           
         
            resk1=0
            p1=0
            rzold1=0
        
          #  print(ksp.shape)
            for c in range(mps.shape[0]):
             
              
                ksp=ksp.detach()
                coord=coord.detach()
                mps=mps.detach()
                dcf=dcf.detach()
                torch.cuda.empty_cache()
           
               
              #  d=d.type(torch.complex64)
                n=0
                if n>=0:
                    k=ksp[c,:,:]
                    co=coord[:,:]
                    d=dcf[:,:]
                
              
               # out0.requires_grad=True
                k=k.cuda()
                co=co.cuda()
                d=d.cuda()
                
                rzold,p,resk=torch.utils.checkpoint.checkpoint(initialize0,out0,k,mps[c],co,d)
                rzold1=rzold1+rzold
                p1=p1+p
                resk1=resk1+resk
            
                torch.cuda.empty_cache()
               


            for j in range(iters):
              
                torch.cuda.empty_cache()
                if n>=0:
                    k=ksp[:,:,:]
                    co=coord[:,:]
                    d=dcf[:,:]
             
             
              
                k=k.cuda()
                co=co.cuda()
                d=d.cuda()
              
                rzold1,p1,out0,resk1=torch.utils.checkpoint.checkpoint(update_CG,out0,k,mps,co,d,rzold1,p1,resk1,alpha)
                loss=0
              

               

    return out0