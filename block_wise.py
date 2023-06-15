import torch
import torch.nn as nn

#neural network
import torch
import numpy as np
from torch.utils import checkpoint
import torch
from gridded_calcs import init_im
import sigpy as sp
#for self-supervision, splits radial spokes into two disjoint subsets
def runner(frac,ksp,dcf,coords,mpsa):
    l = frac
    total_samples = ksp.shape[1]
    selected_samples = int(l * total_samples)

    # Randomly select unique indices for delta
    delta = np.sort(np.random.choice(total_samples, selected_samples, replace=False))
    
    # Get the remaining indices for gamma
    gamma = np.setdiff1d(np.arange(total_samples), delta)

    dcfa = torch.from_numpy(dcf).cuda()
    kspa = torch.from_numpy(ksp).cuda()
    coorda = torch.from_numpy(coords).cuda()

    kgamma = kspa[:, gamma]
    dgamma = dcfa[gamma]
    cgamma = coorda[gamma]
    kdelta = kspa[:, delta]
    ddelta = dcfa[delta]
    cdelta = coorda[delta]

    mpsa_torch = torch.from_numpy(mpsa).cpu()
    ima = checkpoint.checkpoint(init_im, kdelta.cuda(), ddelta.cuda(), cdelta.cuda(), mpsa_torch).detach().cpu().numpy()

    return torch.from_numpy(ima).cuda().unsqueeze(0), kgamma, dgamma, cgamma, kdelta, ddelta, cdelta

def calculate_sense0(M_t,ksp,mps_c,coord_t,dcf):
        torch.cuda.empty_cache()
        r = torch.cuda.memory_reserved(0) /1e9
        F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], torch.reshape(coord_t,[-1,3]), oversamp=1.25, width=4)
        F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
        FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
      
       
        torch.cuda.empty_cache()
       
        torch.cuda.empty_cache()
        e_tc=F_torch.apply(M_t)
        e_tca=torch.complex(e_tc[:,0],e_tc[:,1])
       # e_tca=(e_tca/torch.abs(e_tca).max())*torch.abs(ksp).max()
       # e_tc_update=e_tca
     #   print(torch.abs(e_tca).max())
       # print(torch.abs(ksp).max())
        #loss_self1=torch.nn.MSELoss()
        torch.cuda.empty_cache()
        resk=(((ksp-e_tca)*dcf**0.5))
        torch.cuda.empty_cache()
     
     
        loss=torch.norm(resk,2)**2  #*index_all/ksp.shape[0]
        r = torch.cuda.memory_reserved(0) /1e9
      #  print(r)
        return loss

def SS_update(img_t,ksp_t,dcf_t,coord_t,mpsa): #ima,deform_adjoint1,ksp,coord,dcf,mps,t,device,tr_per_frame):
  
# Data consistency.
  loss_t=0

  
  for c in range(mpsa.shape[0]):
    torch.cuda.empty_cache()
    loss_t=loss_t+torch.utils.checkpoint.checkpoint(calculate_sense0,torch.cat([torch.reshape(torch.real(img_t*mpsa[c].cuda()),[mpsa.shape[1],mpsa.shape[2],mpsa.shape[3],1]),torch.reshape(torch.imag(img_t*mpsa[c].cuda()),[mpsa.shape[1],mpsa.shape[2],mpsa.shape[3],1])],axis=3),torch.reshape(ksp_t[c],[-1]),mpsa[c].cuda(),torch.reshape(coord_t,[-1,3]),torch.reshape(dcf_t,[-1]))
  
  
  return loss_t

import torch.nn as nn
class Net_blockwise(nn.Module):
    def __init__(self,channels):
        super(Net_blockwise, self).__init__()
       # self.conv1b=nn.Conv2d(2,16,[7,1],padding=3,bias=False)
       # self.conv2b=nn.Conv2d(16,2,[1,1],padding=0,bias=False)
        self.conv1 = nn.Conv3d(channels,32, 3,padding=1,bias=False) 
        self.conv2 = nn.Conv3d(32,32,3,padding=1,stride=1,bias=False) #48,28,50
        self.conv3 = nn.Conv3d(32,channels, 3,padding=1,bias=False)
        
        self.conv4 = nn.Conv3d(channels,32 ,3,padding=1,bias=False) 
        self.conv5 = nn.Conv3d(32,32,3,padding=1,stride=1,bias=False) #48,28,50
        self.conv6 = nn.Conv3d(32,channels, 3,padding=1,bias=False)
        
        self.conv7 = nn.Conv3d(channels,32 ,3,padding=1,bias=False) 
        self.conv8 = nn.Conv3d(32,32,3,padding=1,stride=1,bias=False) #48,28,50
        self.conv9 = nn.Conv3d(32,channels, 3,padding=1,bias=False)
        
        self.conv10 = nn.Conv3d(channels,32, 3,padding=1,bias=False) 
        self.conv11 = nn.Conv3d(32,32, 3,padding=1,stride=1,bias=False) #48,28,50
        self.conv12 = nn.Conv3d(32,channels, 3,padding=1,stride=1,bias=False) #48,28,50
        
       
        self.activation=torch.nn.ReLU(inplace=True)
        
        
    def run_function(self,start):
      def custom_forward(input0):
      
        x=self.conv1(input0)
        x=self.activation(x)
        torch.cuda.empty_cache()
        x=self.conv2(x)
        x=self.activation(x)
        out0=self.conv3(x)
        torch.cuda.empty_cache()
        L0=input0+out0
        
        x=self.activation(L0)
        torch.cuda.empty_cache()
        x=self.conv4(x)
        torch.cuda.empty_cache()
       # interp2=torch.cat([out1,interp0],axis=1)
        x=self.activation(x)
        x=self.conv5(x)
        x=self.activation(x)
        out1=self.conv6(x)
        L1=L0+out1
     
        
        x=self.activation(L1)
        x=self.conv7(x)
        x=self.activation(x)
        x=self.conv8(x)
        x=self.activation(x)
        out2=self.conv9(x)
        L2=L1+out2
        
        x=self.activation(L2)
        x=self.conv10(x)
        x=self.activation(x)
        x=self.conv11(x)
        x=self.activation(x)
        out_block=self.conv12(x)
        
        
       
       
       
      
       
       
       
       
       
        return out_block
      return custom_forward
      
    
     
   

   
    def forward(self, vol_input,lo,mps):
      vol_input=vol_input
      if (vol_input.shape[1] % 2) != 0:
            ax=vol_input.shape[1]+1
      else:
            ax=vol_input.shape[1]
      if (vol_input.shape[2] % 2) != 0:
            bx=vol_input.shape[2]+1
      else:
        bx=vol_input.shape[2]
      if (vol_input.shape[3] % 2) != 0:
            cx=vol_input.shape[3]+1
      else:
        cx=vol_input.shape[3]
    
      vol_input_embed=torch.zeros([1,vol_input.shape[0]*2,ax,bx,cx],device='cuda')
      vol_input_real=torch.reshape(torch.real(vol_input),[1,vol_input.shape[0],vol_input.shape[1],vol_input.shape[2],vol_input.shape[3]])
      vol_input_imag=torch.reshape(torch.imag(vol_input),[1,vol_input.shape[0],vol_input.shape[1],vol_input.shape[2],vol_input.shape[3]])
      #vol_inputs=torch.cat([vol_input_real,vol_input_imag],axis=1)
      torch.cuda.empty_cache()
      #vol_input.requires_grad=True
    
      vol_input_embed[:,:vol_input.shape[0],:(vol_input.shape[1]),:(vol_input.shape[2]),:(vol_input.shape[3])]=vol_input_real
      vol_input_embed[:,vol_input.shape[0]:,:(vol_input.shape[1]),:(vol_input.shape[2]),:(vol_input.shape[3])]=vol_input_imag
      torch.cuda.empty_cache()

      
    
      xm=int(ax/2)
      ym=int(bx/2)
      zm=int(cx/2)
      n=0
     
    
  
      out=torch.zeros_like(vol_input_embed[:,:vol_input.shape[0]*2],device='cuda')
      for i in range(0,ax,xm):
            for j in range(0,bx,ym):
                for k in range(0,cx,zm):
               
               
                
                    block=vol_input_embed[:,:,i:xm+i,j:ym+j,k:zm+k]
                    out[:,:,i:xm+i,j:ym+j,k:zm+k]=torch.utils.checkpoint.checkpoint(self.run_function(n),block)
                    
      block_correct=vol_input_embed[:,:,xm-10:xm+10,:,:]
      out_patcha0=torch.utils.checkpoint.checkpoint(self.run_function(n),block_correct)
      block_correct=vol_input_embed[:,:,:,ym-10:ym+10,:]
      out_patchb0=torch.utils.checkpoint.checkpoint(self.run_function(n),block_correct)
      torch.cuda.empty_cache()
      block_correct=vol_input_embed[:,:,:,:,zm-10:zm+10]
      out_patchc0=torch.utils.checkpoint.checkpoint(self.run_function(n),block_correct)
      torch.cuda.empty_cache()
  

        
      out[:,:,xm-5:xm+5,:,:]=out_patcha0[:,:,5:15,:,:]
      out[:,:,:,ym-5:ym+5,:]=out_patchb0[:,:,:,5:15,:]
      out[:,:,:,:,zm-5:zm+5]=out_patchc0[:,:,:,:,5:15]
     
      out=out[:,:,:vol_input.shape[1],:vol_input.shape[2],:vol_input.shape[3]]
      out_final=torch.complex(out[:,:1,:],out[:,1:,:])

      return torch.squeeze(out_final)+torch.squeeze(vol_input[:vol_input.shape[0]]) 






    
      
