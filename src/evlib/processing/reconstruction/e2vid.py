import argparse
import os
import logging
import sys
from typing import Tuple

import numpy as np
import urllib.request

from .e2vid_module.utils.loading_utils import load_model, get_device
from .e2vid_module.image_reconstructor import ImageReconstructor
from .e2vid_module.inference_options import set_inference_options
from ...representation import VoxelGrid  # TODO: import correctly from evlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('ev_lib')

MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  "..", "..", "..", "..", "..", "artifacts",
                                  "E2VID_lightweight.pth.tar")


def progress_bar(count, block_size, total_size):
    completed = count * block_size
    progress = completed / total_size
    bar_length = 50
    filled_length = int(progress * bar_length)
    bar = "=" * filled_length + "-" * (bar_length - filled_length)
    sys.stdout.write(f"\rDownloading: [{bar}] {progress:.2%}")
    sys.stdout.flush()


class E2Vid:
    """Integration of e2vid image reconstruction based on 
       
       https://github.com/uzh-rpg/rpg_vid2e
       
       This wrapper allows you to instantiate the model and
       reconstruct images by using __call__.
       Note that you need to iteratively call this module with subsequent
       event batches for good reconstruction results.
       The number of events per batch can affect the reconstruction results.
    """
    def __init__(self, image_shape: Tuple[int, int], use_gpu: bool = True) -> None:
        """
        Args:
            image_shape: (height, width)
            use_gpu: if GPU should be used for image reconstruction
        """
        assert image_shape[0] > 0
        assert image_shape[1] > 0
        self.image_shape = image_shape

        parser = argparse.ArgumentParser()
        set_inference_options(parser)
        args = parser.parse_args()

        self._download_model()
        self.model = load_model(MODEL_PATH)
        self.device = get_device(use_gpu)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.voxelizer = VoxelGrid(image_shape, self.model.num_bins)
        self.reconstructor = ImageReconstructor(self.model, image_shape[0],
                                                image_shape[1], self.model.num_bins,
                                                args)

    def _download_model(self):
        e2vid_pretrained_url = "http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar"
        
        if not os.path.exists(MODEL_PATH):
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            logger.info(f"Downloading pretrained model file.")
            urllib.request.urlretrieve(e2vid_pretrained_url, filename=MODEL_PATH,
                                       reporthook=progress_bar)
            logger.info("Done.")

    def __call__(self, events:np.ndarray) -> np.ndarray:
        """Reconstruct image from event batch

        Args:
            events: a NumPy array of size [n x d], where n is the number of events and d = 4.
                    Every event is encoded with 4 values (y, x, t, p).

        Returns:
            The reconstructed image of size <image_shape>.
        """
        assert events.shape[1] == 4
        event_tensor = self.voxelizer(events)
        image = self.reconstructor.update_reconstruction(event_tensor)
        return image