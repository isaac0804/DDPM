# Denoising Diffusion Probabilistic Method

Implementation of DDPM in pytorch referencing https://huggingface.co/blog/annotated-diffusion

The backbone will be ConvNeXT for now, but it can be changed to any other architecture.

## Results

| Loss Type                                  | Value |
| -                                          | -     |
| huber loss (1656835168-steadfast-nautilus) | 0.05  |
| L2 loss (1656841502-spiritual-bee)         | 0.12  |
| L1 loss (1656843751-crafty-zebu)           | 0.20  |

| Channel Last    | Ba/s  | per epoch |
| -               | -     | -         |
| Without         | 7.80  | 3:21      |
| With            | 4.10  | 6:18      |
