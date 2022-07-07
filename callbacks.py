import os

import numpy as np
from composer.core.callback import Callback
from matplotlib import animation
from matplotlib import pyplot as plt

class SamplingCallback(Callback):
    def epoch_end(self, state, logger):

        folder = f"runs/{state.run_name}/samples"
        if not os.path.exists(folder):
            os.makedirs(folder)

        fig = plt.figure()
        samples = state.model.sample()
        ims = []
        for i in range(state.model.timesteps):
            samples[i][0] = np.clip(samples[i][0] / 2 + 0.5, 0, 1)
            im = samples[i][0].transpose(1, 2, 0)
            if im.shape[-1] == 3:
                im = plt.imshow(im, animated=True)
            else:
                im = plt.imshow(im, cmap="gray", animated=True)
            ims.append([im])
        animate = animation.ArtistAnimation(
            fig, ims, interval=50, blit=True, repeat_delay=1000)
        folder = f"runs/{state.run_name}/samples"
        animate.save(f"{folder}/sample-{state.timestamp.epoch}.gif")
        plt.close(fig)

class ImplicitSamplingCallback(Callback):
    def epoch_end(self, state, logger):

        folder = f"runs/{state.run_name}/samples"
        if not os.path.exists(folder):
            os.makedirs(folder)

        gens = state.model.sample(num_samples=4).cpu().numpy()
        gens = np.clip(gens / 2 + 0.5, 0, 1)
        for i, gen in enumerate(gens, 0):
            fig = plt.figure()
            ims = []
            for im in gen:
                ims.append([plt.imshow(im, cmap="gray", animated=True)])
            animate = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
            animate.save(f"{folder}/sample-{state.timestamp.epoch}-{i}.gif")
            plt.close(fig)