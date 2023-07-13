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

def calculate_sense0(M_t, ksp, mps_c, coord_t, dcf):
    """
    Function to calculate the loss term, which is the discrepancy between the Fourier-transformed MR image and the original k-space data.
    M_t : Tensor
        Transformed MR Image 
    ksp : Tensor
        Original k-space data
    mps_c : Tensor
        Sensitivity maps
    coord_t : Tensor
        Non-uniform FFT sample locations
    dcf : Tensor
        Density compensation function (dcf)
    """

    # Empty the cache to free up GPU memory
    torch.cuda.empty_cache()

    # Setup NUFFT operator with oversampling ratio 1.25 and kernel width 4
    F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], 
                        torch.reshape(coord_t, [-1, 3]), oversamp=1.25, width=4)

    # Convert NUFFT operators to pytorch functions
    F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
    FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)

    # Apply NUFFT to the transformed image
    e_tc = F_torch.apply(M_t)

    # Separate real and imaginary parts
    e_tca = torch.complex(e_tc[:, 0], e_tc[:, 1])

    # Empty the cache to free up GPU memory
    torch.cuda.empty_cache()

    # Compute residual between NUFFT of image and original k-space data,
    # then weight by the square root of dcf to ensure data consistency
    resk = ((ksp - e_tca) * dcf**0.5)

    # Empty the cache to free up GPU memory
    torch.cuda.empty_cache()

    # Compute the loss as the l2-norm of the residual
    loss = torch.norm(resk, 2)**2

    return loss


def SS_update(img_t, ksp_t, dcf_t, coord_t, mpsa):
    """
    Function to update the image using the calculated loss.
    img_t : Tensor
        Transformed MR Image 
    ksp_t : Tensor
        Original k-space data
    dcf_t : Tensor
        Density compensation function
    coord_t : Tensor
        Non-uniform FFT sample locations
    mpsa : Tensor
        Sensitivity maps
    """
    
    loss_t = 0

    for c in range(mpsa.shape[0]):
        # Free up GPU memory
        torch.cuda.empty_cache()

        # Calculate the loss for each coil and sum them up
        loss_t += torch.utils.checkpoint.checkpoint(
            calculate_sense0,
            torch.cat([torch.reshape(torch.real(img_t * mpsa[c].cuda()), [mpsa.shape[1], mpsa.shape[2], mpsa.shape[3], 1]),
                       torch.reshape(torch.imag(img_t * mpsa[c].cuda()), [mpsa.shape[1], mpsa.shape[2], mpsa.shape[3], 1])], 
                       axis=3),
            torch.reshape(ksp_t[c], [-1]),
            mpsa[c].cuda(),
            torch.reshape(coord_t, [-1, 3]),
            torch.reshape(dcf_t, [-1])
        )

    return loss_t


import torch.nn as nn
class Net_blockwise(nn.Module):
    def __init__(self, channels, block_size):
        super(Net_blockwise, self).__init__()
        self.channels = channels
        self.block_size = block_size

        # Define convolutional layers
        # You can consider moving this to a separate function if the structure repeats
        self.conv_layers = nn.Sequential(
            nn.Conv3d(channels, 32, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, channels, 3, padding=1, bias=False),
        )

    def forward(self, vol_input, lo, mps):
        # Ensure the volume dimensions are even for easier block processing
        ax, bx, cx = [dim if dim % 2 == 0 else dim + 1 for dim in vol_input.shape[1:4]]

        # Initialize output tensor
        out = torch.zeros([1, vol_input.shape[0]*2, ax, bx, cx], device='cuda')

        # Process the volume block by block
        for i in range(0, ax, self.block_size):
            for j in range(0, bx, self.block_size):
                for k in range(0, cx, self.block_size):
                    block = vol_input_embed[:,:,i:self.block_size+i,j:self.block_size+j,k:self.block_size+k]
                    out[:,:,i:self.block_size+i,j:self.block_size+j,k:self.block_size+k] = torch.utils.checkpoint.checkpoint(self.conv_layers, block)

        # Correct padding errors by recomputing the padding region
        padding_correction_regions = [
            (slice(ax//2-10, ax//2+10), slice(None), slice(None)),  # correction for x-axis
            (slice(None), slice(bx//2-10, bx//2+10), slice(None)),  # correction for y-axis
            (slice(None), slice(None), slice(cx//2-10, cx//2+10))  # correction for z-axis
        ]
        for region in padding_correction_regions:
            block = vol_input_embed[:,:,region[0],region[1],region[2]]
            out_patch = torch.utils.checkpoint.checkpoint(self.conv_layers, block)
            out[:,:,region[0],region[1],region[2]] = out_patch

        # Trim output to match the original volume shape
        out = out[:,:,:vol_input.shape[1],:vol_input.shape[2],:vol_input.shape[3]]
        out_final = torch.complex(out[:,:1,:],out[:,1:,:])

        return torch.squeeze(out_final) + torch.squeeze(vol_input[:vol_input.shape[0]])







    
      
