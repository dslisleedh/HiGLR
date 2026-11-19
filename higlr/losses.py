"""Five-scale RGB MS-SSIM.

Wang et al. 2003 Eq. 7, weights from section 3. Same zero padding makes
the 11-tap Gaussian defined even at the 8x8 last scale of PT128 crops.
This boundary choice and the outer gamma are not claimed as ClusIR settings.
"""
import torch
from torch.nn import functional as F

SETTINGS = dict(scales=5, weights=[.0448,.2856,.3001,.2363,.1333],
                window_size=11, sigma=1.5, data_range=1., K=[.01,.03],
                padding='same_zero', reduction='spatial mean then scale product then RGB/batch mean',
                stability_floor=1e-8, downsample='avg_pool2d kernel2 stride2')


def ms_ssim(x,y):
    if x.shape != y.shape or x.ndim != 4 or min(x.shape[-2:]) < 16:
        raise ValueError('MS-SSIM expects matching NCHW images, spatial size >=16')
    if x.dtype not in (torch.float32,torch.float64) or x.dtype != y.dtype:
        raise ValueError('MS-SSIM requires matching FP32/FP64 inputs')
    c=x.shape[1]
    axis=torch.arange(-5,6,device=x.device,dtype=x.dtype)
    g=torch.exp(-axis.square()/(2*1.5**2));g=g/g.sum()
    horizontal=g.view(1,1,1,11).expand(c,1,1,11).contiguous()
    vertical=g.view(1,1,11,1).expand(c,1,11,1).contiguous()
    def blur(z):
        return F.conv2d(F.conv2d(z,horizontal,padding=(0,5),groups=c),vertical,padding=(5,0),groups=c)
    terms=[]
    for level in range(5):
        mx,my=blur(x),blur(y)
        vx,vy=blur(x*x)-mx*mx,blur(y*y)-my*my
        cov=blur(x*y)-mx*my
        cs=(2*cov+.03**2)/(vx+vy+.03**2)
        value=cs if level<4 else cs*(2*mx*my+.01**2)/(mx*mx+my*my+.01**2)
        terms.append(value.mean((-2,-1)).clamp_min(1e-8))
        if level<4:x,y=F.avg_pool2d(x,2),F.avg_pool2d(y,2)
    values=torch.stack(terms)
    weights=x.new_tensor(SETTINGS['weights']).view(5,1,1)
    return values.pow(weights).prod(0).mean()


def restoration_loss(restored, target, scores, labels, config):
    restored = restored.clamp(0, 1)
    pixel = F.l1_loss(restored, target)
    frequency = F.l1_loss(torch.view_as_real(torch.fft.fft2(restored)),
                         torch.view_as_real(torch.fft.fft2(target)))
    structural = 1 - ms_ssim(restored, target)
    classification = F.cross_entropy(scores, labels, label_smoothing=config['label_smoothing'])
    total = (config['l1'] * pixel + config['fft'] * frequency
             + config['msssim'] * structural + config['ce'] * classification)
    return total

