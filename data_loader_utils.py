import h5py
import numpy as np
import logging
import sigpy
import sigpy as sp
import cupy
import sigpy.mri as mr
import os

def load_data(MR_Raw,RO_factor):

    kdata,coords,dcfs,resps,tr=convert_ute(MR_Raw, max_coils=20, dsfSpokes=1.0, compress_coils=False) #unpack data

    coords=autofova(coords, dcfs,kdata,device=None,
                thresh=0.15, scale=1, oversample=2.0) #resize FOV
    RO=int(kdata.shape[2]/RO_factor) #readout for center out allows you to choose your spatial resolution, currently I just set it to full spatial resolution
    kdata=kdata[:,:,:RO]
    coords=coords[:,:RO]
    dcfs=dcfs[:,:RO]
    mpsa=mr.app.JsenseRecon(kdata, coord=coords, weights=dcfs, device=0).run() #generate sense maps
    tr_per_frame=kdata.shape[1]
    #normalize data
    dcf=normalize(mpsa,coords,dcfs,kdata,tr_per_frame)
    ksp=kspace_scaling(mpsa,dcf,coords,kdata)
    return ksp,coords,dcf,mpsa



#sorts breath hold data through time function
def convert_ute(h5_file, max_coils=20, dsfSpokes=1.0, compress_coils=False):
    with h5py.File(h5_file, "r") as hf:
        try:
            num_encodes = np.squeeze(hf["Kdata"].attrs["Num_Encodings"])
            num_coils = np.squeeze(hf["Kdata"].attrs["Num_Coils"])
            num_frames = np.squeeze(hf["Kdata"].attrs["Num_Frames"])
            trajectory_type = [
                np.squeeze(hf["Kdata"].attrs["trajectory_typeX"]),
                np.squeeze(hf["Kdata"].attrs["trajectory_typeY"]),
                np.squeeze(hf["Kdata"].attrs["trajectory_typeZ"]),
            ]
            dft_needed = [
                np.squeeze(hf["Kdata"].attrs["dft_neededX"]),
                np.squeeze(hf["Kdata"].attrs["dft_neededY"]),
                np.squeeze(hf["Kdata"].attrs["dft_neededZ"]),
            ]
            logging.info(f"Frames {num_frames}")
            logging.info(f"Coils {num_coils}")
            logging.info(f"Encodings {num_encodes}")
            logging.info(f"Trajectory Type {trajectory_type}")
            logging.info(f"DFT Needed {dft_needed}")
        except Exception:
            logging.info("Missing H5 Attributes...")
            num_coils = 0
            while f"KData_E0_C{num_coils}" in hf["Kdata"]:
                num_coils += 1
            logging.info(f"Number of coils: {num_coils}")
            num_encodes = 0
            while f"KData_E{num_encodes}_C0" in hf["Kdata"]:
                num_encodes += 1
            logging.info(f"Number of encodes: {num_encodes}")
        # if max_coils is not None:
        #     num_coils = min(max_coils, num_coils)
        coords = []
        dcfs = []
        kdata = []
        ecgs = []
        resps = []
        print('test')
        for encode in range(num_encodes):
            print(num_encodes)
           
            try:
                time = np.squeeze(hf["Gating"][f"time"])
                order = np.argsort(time)
            except Exception:
                time = np.squeeze(hf["Gating"][f"TIME_E{encode}"])
                order = np.argsort(time)
            try:
                resp = np.squeeze(hf["Gating"][f"resp"])
                resp = resp[order]
            except Exception:
                resp = np.squeeze(hf["Gating"][f"RESP_E{encode}"])
                resp = resp[order]
            print('coord')
            coord = []
            for i in ["Z", "Y", "X"]:
                # logging.info(f"Loading {i} coords.")
                coord.append(hf["Kdata"][f"K{i}_E{encode}"][0][order])
            coord = np.stack(coord, axis=-1)
           
            dcf = np.array(hf["Kdata"][f"KW_E{encode}"][0][order])
          
            try:
                ecg = np.squeeze(hf["Gating"][f"ecg"])
                ecg = ecg[order]
            except Exception:
                ecg = np.squeeze(hf["Gating"][f"ECG_E{encode}"])
                ecg = ecg[order]
            # Get k-space
            ksp = []
            print('coils')
            for c in range(num_coils):
                print(c)
                ksp.append(
                    hf["Kdata"][f"KData_E{encode}_C{c}"]["real"][0][order]
                    + 1j * hf["Kdata"][f"KData_E{encode}_C{c}"]["imag"][0][order]
                )
          
            ksp = np.stack(ksp, axis=0)
            logging.info("num_coils {}".format(num_coils))
            import sigpy.mri as mr
            print('noisy')
           
          
           # noise = hf["Kdata"]["Noise"]["real"] + 1j * hf["Kdata"]["Noise"]["imag"]
           # logging.info("Whitening ksp.")
           # cov = mr.get_cov(noise)
           # ksp = mr.whiten(ksp, cov)
          
           
           # logging.warning(f"{err}. Scaling k-space by max value.")
            ksp /= np.abs(ksp).max()
            if compress_coils:
                logging.info("Compressing to {} channels.".format(max_coils))
                ksp = pca_cc(kdata=ksp, axis=0, target_channels=max_coils)
            # Append to list
            coords.append(coord)
            dcfs.append(dcf)
            kdata.append(ksp)
            ecgs.append(ecg)
            resps.append(resp)
         
    # Stack the data along projections (no reason not to keep encodes separate in my case)
    print('stack')
    kdata = np.concatenate(kdata, axis=1)
    dcfs = np.concatenate(dcfs, axis=0)
    coords = np.concatenate(coords, axis=0)
    ecgs = np.concatenate(ecgs, axis=0)
    resps = np.concatenate(resps, axis=0)
    # crop empty calibration region (1800 spokes for ipf)
    kdata = kdata[:, :, :]
    coords = coords[:, :, :]
    dcfs = dcfs[:, :]
    resps = resps[:]
    ecgs = ecgs[:]
    # crop to desired number of spokes (all by default)
    totalSpokes = kdata.shape[1]
    nSpokes = int(totalSpokes // dsfSpokes)
    kdata = kdata[:, :nSpokes, :]
    coords = coords[:nSpokes, :, :]
    dcfs = dcfs[:nSpokes, :]
    resps = resps[:nSpokes]
    print('complete')
   
    # Get TR
    d_time = time[order]
    tr = d_time[1] - d_time[0]
    return kdata, coords, dcfs, resps / resps.max(), tr

#normalize breath hold data
def normalize(mps,coord,dcf,ksp,tr_per_frame):
    mps=mps
   
   # import cupy
    import sigpy as sp
    device=0
    # Estimate maximum eigenvalue.
    coord_t = sp.to_device(coord[:tr_per_frame], device)
    dcf_t = sp.to_device(dcf[:tr_per_frame], device)
    F = sp.linop.NUFFT([mps.shape[1],mps.shape[2],mps.shape[3]], coord_t)
    W = sp.linop.Multiply(F.oshape, dcf_t)

    max_eig = sp.app.MaxEig(F.H * W * F, max_iter=500, device=0,
                            dtype=ksp.dtype,show_pbar=True).run()
    dcf1=dcf/max_eig
    return dcf1
#same as above
def kspace_scaling(mps,dcf,coord,ksp):
    # Estimate scaling.
    img_adj = 0
    device=0
    dcf = sp.to_device(dcf, device)
    coord = sp.to_device(coord, device)
   
    for c in range(mps.shape[0]):
        print(c)
        mps_c = sp.to_device(mps[c], device)
        ksp_c = sp.to_device(ksp[c], device)
        img_adj_c = sp.nufft_adjoint(ksp_c * dcf, coord, [mps.shape[1],mps.shape[2],mps.shape[3]])
        img_adj_c *= cupy.conj(mps_c)
        img_adj += img_adj_c


    img_adj_norm = cupy.linalg.norm(img_adj).item()
    print(img_adj_norm)
    ksp1=ksp/img_adj_norm
    return ksp1

#resize coords (k-space trajectory) so that all imaged body fits in the FOV 
def autofova(coords, dcf,kdata,device=None,
            thresh=0.15, scale=1, oversample=2.0):
    #logger = logging.getLogger('autofov')
    import cupy as xp
    # Set to GPU
    if device is None:
        device = sp.Device(0)
  #  logger.info(f'Device = {device}')
    xp = device.xp

    with device:
        # Put on GPU
        coord = 2.0 * coords
       # dcf = mri_raw.dcf[0]

        # Low resolution filter
        res = 64
        lpf = np.sum(coord ** oversample, axis=-1)
        lpf = np.exp(-lpf / (2.0 * res * res))

        # Get reconstructed size
        img_shape = sp.estimate_shape(coord)
        img_shape = [int(min(i, 64)) for i in img_shape]
        images = xp.ones([20] + img_shape, dtype=xp.complex64)
        kdata = kdata

        sos = xp.zeros(img_shape, dtype=xp.float32)

      #  logger.info(f'Kdata shape = {kdata[0].shape}')
      #  logger.info(f'Images shape = {images.shape}')

        coord_gpu = sp.to_device(coord, device=device) # coord needs to be push to device in new sigpy version

        for c in range(kdata.shape[0]):
            print(c)
        #    logger.info(f'Reconstructing  coil {c}')
            ksp_t = np.copy(kdata[c, ...])
            ksp_t *= np.squeeze(dcf)
            ksp_t *= np.squeeze(lpf)
            ksp_t = sp.to_device(ksp_t, device=device)

            sos += xp.square(xp.abs(sp.nufft_adjoint(ksp_t, coord_gpu, img_shape)))

        # Multiply by SOS of low resolution maps
        sos = xp.sqrt(sos)

    sos = sp.to_device(sos)


    # Spherical mask
    zz, xx, yy = np.meshgrid(np.linspace(-1, 1, sos.shape[0]),
                             np.linspace(-1, 1, sos.shape[1]),
                             np.linspace(-1, 1, sos.shape[2]),indexing='ij')
    rad = zz ** 2 + xx ** 2 + yy ** 2
    idx = ( rad >= 1.0)
    sos[idx] = 0.0

    # Export to file
    out_name = 'AutoFOV.h5'
   # logger.info('Saving autofov to ' + out_name)
    try:
        os.remove(out_name)
    except OSError:
        pass
    with h5py.File(out_name, 'w') as hf:
        hf.create_dataset("IMAGE", data=np.abs(sos))

    boxc = sos > thresh * sos.max()
    boxc_idx = np.nonzero(boxc)
    boxc_center = np.array(img_shape) // 2
    boxc_shape = np.array([int(2 * max(c - min(boxc_idx[i]), max(boxc_idx[i]) - c) * scale)
                           for i, c in zip(range(3), boxc_center)])

    #  Due to double FOV scale by 2
    target_recon_scale = boxc_shape / img_shape
   # logger.info(f'Target recon scale: {target_recon_scale}')

    # Scale to new FOV
    target_recon_size = sp.estimate_shape(coord) * target_recon_scale

    # Round to 16 for blocks and FFT
    target_recon_size = 16*np.ceil( target_recon_size / 16 )

    # Get the actual scale without rounding
    ndim = coord.shape[-1]

    with sp.get_device(coord):
        img_scale = [(2.0*target_recon_size[i]/(coord[..., i].max() - coord[..., i].min())) for i in range(ndim)]

    # fix precision errors in x dir
    for i in range(ndim):
        round_img_scale = round(img_scale[i], 6)
        if round_img_scale - img_scale[i] < 0:
            round_img_scale += 0.000001
        img_scale[i] = round_img_scale

  #  logger.info(f'Target recon size: {target_recon_size}')
  #  logger.info(f'Kspace Scale: {img_scale}')

    for e in range(len(coord)):
        coords[e] *= img_scale

    new_img_shape = sp.estimate_shape(coords)
    print(sp.estimate_shape(coords))
    #print(sp.estimate_shape(mri_raw.coords[1]))
    #print(sp.estimate_shape(mri_raw.coords[2]))
    #print(sp.estimate_shape(mri_raw.coords[3]))
    #print(sp.estimate_shape(mri_raw.coords[4]))

   # logger.info('Image shape: {}'.format(new_img_shape))
    return coords
    
    