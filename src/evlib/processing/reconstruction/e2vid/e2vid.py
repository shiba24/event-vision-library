import os
import logging
import sys

import numpy as np
import urllib.request

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('ev_lib')


def progress_bar(count, block_size, total_size):
    completed = count * block_size
    progress = completed / total_size
    bar_length = 50
    filled_length = int(progress * bar_length)
    bar = "=" * filled_length + "-" * (bar_length - filled_length)
    sys.stdout.write(f"\rDownloading: [{bar}] {progress:.2%}")
    sys.stdout.flush()


class E2Vid:
    def __init__(self) -> None:
        self._load_model()

        model = load_model(args.path_to_model)
        device = get_device(args.use_gpu)

        model = model.to(device)
model.eval()

    def _load_model(self):
        e2vid_pretrained_url = "http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar"

        pretrained_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  "..", "..", "..", "..", "..", "artifacts",
                                  "E2VID_lightweight.pth.tar")
        
        if not os.path.exists(pretrained_path):
            os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)
            logger.info(f"Downloading pretrained model file.")
            urllib.request.urlretrieve(e2vid_pretrained_url, filename=pretrained_path,
                                       reporthook=progress_bar)
            logger.info("Done.")

    def __call__(self, ) -> np.ndarray:
        image = np.zeros((1, 1), dtype=int)
        return image