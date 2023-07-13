# Import necessary libraries
import torch
import sigpy as sp

def initialize(M_ta, ksp, mps_c, coord_t, dcf):
    """ 
    Set up initial conditions for conjugate gradient descent
    """
    # Set up NUFFT operator
    F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], coord_t.detach().cpu().numpy(), oversamp=1.25, width=4)
    F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
    FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
      
    # Prepare k-space data by applying density compensation
    k = (ksp) * dcf
    k_com = torch.cat([torch.real(torch.squeeze(k)).unsqueeze(axis=2), torch.imag(torch.squeeze(k)).unsqueeze(axis=2)], axis=2)

    # Compute initial guess for solution
    g = FH_torch.apply(k_com)
    y_guess = torch.complex(g[:,:,:,0],g[:,:,:,1])*torch.conj(mps_c).cuda()

    # Calculate residual
    Ap0 = torch.utils.checkpoint.checkpoint(operator, M_ta, mps_c, dcf, coord_t)
    resk = y_guess - Ap0
    p = resk
    rzold = torch.real(torch.vdot(resk.flatten(), resk.flatten()))
    
    # Clean up GPU memory
    torch.cuda.empty_cache()
    
    return rzold, p, resk

def update_CG(M_ta, ksp, mps, coord_t, dcf, rzold, p, resk):
    """
    Perform one iteration of conjugate gradient descent
    """
    # Set up NUFFT operator
    F = sp.linop.NUFFT([mps.shape[1], mps.shape[2], mps.shape[3]], coord_t.detach().cpu().numpy(), oversamp=1.25, width=4)
    F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
    FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)

    # Compute Ap for each coil and sum them up
    Ap = sum(torch.utils.checkpoint.checkpoint(operator, p, mps[c], dcf, coord_t) for c in range(mps.shape[0]))

    # Update p and calculate new residual
    pAp = torch.real(torch.vdot(p.flatten(), Ap.flatten()))
    alpha = (rzold / pAp).cuda()
    M_ta = M_ta + alpha * p
    resk = resk - alpha * Ap
    rznew = torch.real(torch.vdot(resk.flatten(), resk.flatten()))
    beta = (rznew / rzold).cuda()
    p = resk + beta * p

    # Clean up GPU memory
    torch.cuda.empty_cache()

    return rznew, p, M_ta, resk

def operator(x, mps_c, dcf, coord_t):
    """
    Perform a forward and backward transformation between image and k-space
    """
    # Set up NUFFT operator
    F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], coord_t.detach().cpu().numpy(), oversamp=1.25, width=4)
    F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
    FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)

    # Forward operation: from image space to k-space
    diff = torch.cat([torch.reshape(torch.real(x*mps_c),[mps_c.shape[0],mps_c.shape[1],mps_c.shape[2],1]),
                      torch.reshape(torch.imag(x*mps_c),[mps_c.shape[0],mps_c.shape[1],mps_c.shape[2],1])], axis=3)
    e_tc = F_torch.apply(diff)
    e_tca = torch.complex(e_tc[:,:,0], e_tc[:,:,1]) * dcf

    # Backward operation: from k-space to image space
    e_tca_com = torch.cat([torch.real(torch.squeeze(e_tca)).unsqueeze(axis=2), torch.imag(torch.squeeze(e_tca)).unsqueeze(axis=2)], axis=2)
    g = FH_torch.apply(e_tca_com)
    gout = torch.complex(g[:,:,:,0], g[:,:,:,1]) * torch.conj(mps_c).cuda()
   
    return gout 

def CG_DClayer(out0, ksp, coord, dcf, mps, iters):
    """
    Perform CG iterations
    """
    # Initializations
    resk1, p1, rzold1 = 0, 0, 0

    # Calculate initial residuals, search directions, and old residuals
    for c in range(mps.shape[0]):
        k = ksp[c,:,:].cuda()
        co = coord[:,:].cuda()
        d = dcf[:,:].cuda()
        rzold, p, resk = torch.utils.checkpoint.checkpoint(initialize, out0, k, mps[c].cuda(), co, d)
        rzold1 += rzold
        p1 += p
        resk1 += resk
        torch.cuda.empty_cache()

    # Perform CG iterations
    for _ in range(iters):
        k = ksp[:,:,:].cuda()
        co = coord[:,:].cuda()
        d = dcf[:,:].cuda()
        rzold1, p1, out0, resk1 = torch.utils.checkpoint.checkpoint(update_CG, out0, k, mps.cuda(), co, d, rzold1, p1, resk1)
        torch.cuda.empty_cache()

    return out0
