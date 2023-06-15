def calculate_sense0a(M_t,ksp,mps_c,coord_t,dcf):
        
        F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], coord_t, oversamp=1.25, width=4)
        F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
        FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
        e_tc=F_torch.apply(M_t)
        e_tca=torch.complex(e_tc[:,0],e_tc[:,1])
     
  
       
        resk=(e_tca-ksp)*dcf
        resk_real=torch.real(torch.squeeze(resk))
        resk_imag=torch.imag(torch.squeeze(resk))
    
        resk_real=resk_real.unsqueeze(axis=1)
        resk_imag=resk_imag.unsqueeze(axis=1)
        resk_com=torch.cat([resk_real,resk_imag],axis=1)
        g=torch.complex(g[:,:,:,0],g[:,:,:,1])*torch.conj(mps_c)
        return g.cuda()

def grad_step(img_t,ksp_t,dcf_t,coord_t,mps,alpha): #ima,deform_adjoint1,ksp,coord,dcf,mps,t,device,tr_per_frame):
 

# Data consistency.

  g=0
  for c in range(mps.shape[0]):
 
 

    g=g+torch.utils.checkpoint.checkpoint(calculate_sense0a,torch.cat([torch.reshape(torch.real(img_t*mps[c].cuda()),[mps.shape[1],mps.shape[2],mps.shape[3],1]),torch.reshape(torch.imag(img_t*mps[c].cuda()),[mps.shape[1],mps.shape[2],mps.shape[3],1])],axis=3),torch.reshape(ksp_t[c],[-1]),mps[c].cuda(),torch.reshape(coord_t,[-1,3]),torch.reshape(dcf_t,[-1]))
  img_t=img_t-alpha*g
 





  return img_t
