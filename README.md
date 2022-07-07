# Denoising Diffusion Probabilistic Method

Implementation of DDPM in pytorch referencing [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion).

The backbone will be ConvNeXT for now, but it can be changed to any other architecture.

## Results

![Gray sample after 30 epochs](./assets/sample-30ep.gif)
![Color sample after 100 epochs](./assets/sample-100ep.gif)

## Hyperparameters

Timesteps: 200
Batch Size: 64

## Bugs along the way

- Tensorboard logger can't be used as the timeout value is too large.
- Autoresume on the first epoch is not working.
- Channel last reduce overall speed
    | Channel Last    | Ba/s  | per epoch |
    | -               | -     | -         |
    | Without         | 7.80  | 3:21      |
    | With            | 4.10  | 6:18      |
