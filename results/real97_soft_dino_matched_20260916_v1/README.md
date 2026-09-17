# Soft matched metrics: Standard, Ours and DINO

Provisional object tracking; image scores are complete. No action metric.
Common future: native frames 3,6,...,84; no condition frames or padding.

## test_all9

| Method | PSNR | SSIM | LPIPS | Common ADE | Final FDE | Pair coverage |
|---|---:|---:|---:|---:|---:|---:|
| standard | 33.09998 | 0.96439 | 0.07750 | 30.28475 | 48.75180 | 0.99405 |
| stage1 | 32.34666 | 0.96067 | 0.07932 | 22.52553 | 40.67482 | 0.99802 |
| ours | 32.46726 | 0.96077 | 0.07926 | 22.69350 | 40.34636 | 0.98810 |
| dino | 33.45580 | 0.96505 | 0.07148 | 22.34220 | 36.48137 | 0.99603 |

## test_shared6

| Method | PSNR | SSIM | LPIPS | Common ADE | Final FDE | Pair coverage |
|---|---:|---:|---:|---:|---:|---:|
| standard | 32.90366 | 0.96421 | 0.07958 | 32.78484 | 39.41445 | 0.99702 |
| stage1 | 32.51668 | 0.96138 | 0.07536 | 17.69410 | 22.66169 | 1.00000 |
| ours | 32.61229 | 0.96146 | 0.07555 | 18.09545 | 22.37871 | 0.99405 |
| dino | 33.47200 | 0.96539 | 0.07087 | 21.19066 | 29.16508 | 0.99702 |

## train_all9

| Method | PSNR | SSIM | LPIPS | Common ADE | Final FDE | Pair coverage |
|---|---:|---:|---:|---:|---:|---:|
| standard | 35.40585 | 0.96976 | 0.05496 | 7.58614 | 10.56250 | 1.00000 |
| stage1 | 34.19170 | 0.96612 | 0.06086 | 7.24409 | 10.90578 | 0.99206 |
| ours | 34.29781 | 0.96621 | 0.06071 | 7.30779 | 10.93393 | 0.99603 |
| dino | 36.69406 | 0.97415 | 0.04608 | 2.73721 | 4.08967 | 0.99008 |

## Comparability

- Standard trained only six of these environments and uses five repeated initial frames.
- Ours uses known-family mean initialization; DINO uses four train supports.
- ADE common mask is recomputed including DINO, not the previous global-stage2 mask.
- Missing detections are not zero error; FDE requires actual final eligible frame.
