# Decode-mode profiling summary

Sources: 5 CSV files, 600 rows, 5 GPUs, 3 models.

## Per-GPU × per-model decode p50 (ms)

```
                                   count    min  median    max
actual_gpu_name       model_name                              
NVIDIA A10            gpt2-large      40  17.28   17.89  20.65
                      gpt2-medium     40  11.53   12.25  12.89
                      gpt2-small      40   5.91    6.20   6.37
NVIDIA A100-SXM4-40GB gpt2-large      40  26.50   28.61  29.33
                      gpt2-medium     40  18.42   18.81  19.48
                      gpt2-small      40   9.79   10.67  15.07
NVIDIA B200           gpt2-large      40   6.52    7.04  11.50
                      gpt2-medium     40   4.30    4.72   6.63
                      gpt2-small      40   2.22    2.38   2.83
NVIDIA H100 80GB HBM3 gpt2-large      40  15.42   16.31  17.36
                      gpt2-medium     40   9.65   11.00  11.33
                      gpt2-small      40   4.37    5.70   5.84
NVIDIA L4             gpt2-large      40  20.26   21.75  32.60
                      gpt2-medium     40  13.54   14.66  16.02
                      gpt2-small      40   6.95    7.50   7.66
```

## Cross-GPU comparison @ batch=1, seq=128

```
model_name             gpt2-large  gpt2-medium  gpt2-small
actual_gpu_name                                           
NVIDIA A10                  17.85        11.65        6.02
NVIDIA A100-SXM4-40GB       27.32        19.48       10.38
NVIDIA B200                  7.03         4.55        2.25
NVIDIA H100 80GB HBM3       15.88        10.67        4.58
NVIDIA L4                   21.52        14.00        7.40
```
