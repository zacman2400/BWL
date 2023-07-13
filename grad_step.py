# Required packages
import torch
import scipy as sp

def calculate_sense0a(input_image, kspace_data, sensitivity_map, coordinates, density_compensation):
    """
    This function calculates the sense of a given image using Non-uniform FFT and gradients.
    
    Parameters:
    input_image: The input image data
    kspace_data: The k-space data
    sensitivity_map: The coil sensitivity map
    coordinates: The coordinates for Non-uniform FFT
    density_compensation: The density compensation factor
    """
        
    # Compute Non-uniform FFT
    non_uniform_fft = sp.linop.NUFFT([sensitivity_map.shape[0], sensitivity_map.shape[1], sensitivity_map.shape[2]], coordinates, oversamp=1.25, width=4)
    nufft_torch = sp.to_pytorch_function(non_uniform_fft, input_iscomplex=True, output_iscomplex=True)
    nufft_torch_adj = sp.to_pytorch_function(non_uniform_fft.H, input_iscomplex=True, output_iscomplex=True)
    
    encoded_image = nufft_torch.apply(input_image)
    encoded_image_complex = torch.complex(encoded_image[:,0],encoded_image[:,1])
    
    # Compute residuals
    residuals = (encoded_image_complex - kspace_data) * density_compensation
    residuals_real = torch.real(torch.squeeze(residuals))
    residuals_imag = torch.imag(torch.squeeze(residuals))
    
    residuals_real = residuals_real.unsqueeze(axis=1)
    residuals_imag = residuals_imag.unsqueeze(axis=1)
    residuals_complex = torch.cat([residuals_real, residuals_imag], axis=1)
    
    gradient = torch.complex(residuals_complex[:,:,:,0], residuals_complex[:,:,:,1]) * torch.conj(sensitivity_map)
    
    return gradient.cuda()

def gradient_step(image, kspace, density_compensation, coordinates, sensitivity_map, learning_rate): 
    """
    This function performs a gradient step in the image reconstruction process.
    
    Parameters:
    image: The input image data
    kspace: The k-space data
    density_compensation: The density compensation factor
    coordinates: The coordinates for Non-uniform FFT
    sensitivity_map: The coil sensitivity map
    learning_rate: The learning rate for the gradient step
    """
    # Initialize gradient
    gradient = 0
    for coil in range(sensitivity_map.shape[0]):
        reshaped_real_image = torch.reshape(torch.real(image*sensitivity_map[coil].cuda()),[sensitivity_map.shape[1], sensitivity_map.shape[2], sensitivity_map.shape[3],1])
        reshaped_imag_image = torch.reshape(torch.imag(image*sensitivity_map[coil].cuda()),[sensitivity_map.shape[1], sensitivity_map.shape[2], sensitivity_map.shape[3],1])
        concatenated_image = torch.cat([reshaped_real_image, reshaped_imag_image], axis=3)

        gradient += torch.utils.checkpoint.checkpoint(calculate_sense0a, concatenated_image, torch.reshape(kspace[coil],[-1]), sensitivity_map[coil].cuda(), torch.reshape(coordinates,[-1,3]), torch.reshape(density_compensation,[-1]))

    # Update image using gradient
    image = image - learning_rate * gradient

    return image

