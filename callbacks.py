import os

import numpy as np
from composer.core.callback import Callback
from matplotlib import animation
from matplotlib import pyplot as plt
from regex import P

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

cifar_10_class = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]