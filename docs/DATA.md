# Data and implementation defaults

The manuscript specifies the optimizer, losses, stage schedules, batch size and EMA. The table also lists implementation choices for data handling and initialization.

For AIO-3/5, `--data-root` points to a directory containing `Denoise/`, `Derain/`, `Dehaze/` and, for AIO-5, `Deblur/`, `Lowlight/`. These follow the layouts in the [pinned R2R README](../reference/r2r/README.md). Supply alternative filename lists with `--data-lists /path/to/data_dir`; by default the [upstream lists](../reference/r2r/data_dir) are used. Files and paired geometry must already exist; no data download or copying is performed.

For CDD-11, the root contains `train/clear/` and `train/{haze,haze_rain,haze_snow,low,low_haze,low_haze_rain,low_haze_snow,low_rain,low_snow,rain,snow}/`. Each folder has the same 1,183 PNG filenames. Test data is not loaded by the trainer.

| Choice | Implementation |
|---|---|
| Seed | 1415926; rank-specific augmentation RNG |
| AIO sampling | R2R lists/repetitions: noise×9 (three sigma lists×3); AIO3 rain×120/haze×1; AIO5 rain×80/haze×1/blur×30/lowlight×60 |
| AIO list order | Canonical sort before epoch shuffling, avoiding upstream filesystem-order dependence |
| AIO label IDs | Noise sigma15/25/50→class0, rain→1, haze→2, blur→3, lowlight→4 |
| CDD sampling | Uniform degradation then uniform image with replacement; list×2, logical length26,026; sorted 11-class order |
| Augmentation | Identity-inclusive dihedral transforms; rain and CDD rain/snow composites use modes0/1/4/5 to preserve their axis |
| AIO crop | R2R base16 center trim; paired random crop. At FT256, short inputs are resized with shared bicubic geometry before crop |
| Noise | Generated after crop/augmentation; Gaussian sigma15/25/50 in 8-bit space, clip and uint8 quantization, then convert to [0,1] |
| CDD crop | Aligned random crop; undersized images raise an error |
| Restoration loss | Predictions clamped to [0,1] before L1, FFT and MS-SSIM; FFT is unnormalized FFT2 with real/imaginary L1 |
| MS-SSIM | Five scales, 11-tap sigma1.5 Gaussian, same zero padding, standard scale weights, data range1 |
| Initialization | Default random RGB weight, bias0.5; residual scales0.01; 8×8 Q/K/V and degradation summarizers initialized to 1/64 with zero bias where present |
| Gradient clipping | None |
| Checkpoints | latest/previous every epoch, PT/CDD10-epoch and FT5-epoch milestones, final always retained |

The CDD sampler follows the established independently implemented HiGLR CDD adapter, whose provenance is recorded in [cdd11_data.py](../higlr/cdd11_data.py). It is not silently replaced by R2R's compos loader. New run initialization and dataset ordering are reproducible within this repository's protocol, but are not claimed to reproduce historical runs bit for bit.
