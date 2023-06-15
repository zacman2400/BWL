#self-supervised k-space subset selection if using self-supervised loss
def diff(first, second):
            second = set(second)
            return [item for item in first if item not in second]

def runner(frac):
    import random
    import torch
    from torch.utils import checkpoint
    l=frac 
   
   
    delta=np.sort(random.sample(range(0, ksp.shape[1]), int(l*ksp.shape[1])))
    R=np.arange(0,ksp.shape[1])
   
    gamma=diff(R,delta)
    gamma=np.array(gamma)
    dcfa=torch.from_numpy(dcf).cuda()
    kspa=torch.from_numpy(ksp).cuda()
    coorda=torch.from_numpy(coords).cuda()

    kgamma=kspa[:,gamma]
    dgamma=dcfa[gamma] #*index_large.max()/(kspga.shape[1]*.6)
    cgamma=coorda[gamma]
    kdelta=kspa[:,delta]
    ddelta=dcfa[delta] #*index_large.max()/(kspga.shape[1]*.4)
    cdelta=coorda[delta]
    ima=torch.utils.checkpoint.checkpoint(init_im,kdelta.cuda(),ddelta.cuda(),cdelta.cuda(),torch.from_numpy(mpsa).cpu()).detach().cpu().numpy()
    return torch.from_numpy(ima).cuda().unsqueeze(0),kgamma,dgamma,cgamma,kdelta,ddelta,cdelta