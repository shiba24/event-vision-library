from typing import Any, Union

import torch
import numpy as np

from .model.model import *
from .utils.inference_utils import CropParameters, EventPreprocessor, IntensityRescaler, ImageFilter, UnsharpMaskFilter
from .utils.loading_utils import get_device

class ImageReconstructor:
    def __init__(self, model: torch.nn.Module, height: int, width: int,
                 num_bins: int, options: Any) -> None:
        self.model = model
        self.use_gpu = options.use_gpu
        self.device = get_device(self.use_gpu)
        self.height = height
        self.width = width
        self.num_bins = num_bins

        self.initialize(self.height, self.width, options)

    def initialize(self, height: int, width: int, options: Any) -> None:
        print('== Image reconstruction == ')
        print('Image size: {}x{}'.format(self.height, self.width))

        self.no_recurrent = options.no_recurrent
        if self.no_recurrent:
            print('!!Recurrent connection disabled!!')

        n_encoders: int = self.model.num_encoders  # type: ignore
        self.crop = CropParameters(self.width, self.height, n_encoders)

        self.last_states_for_each_channel = {'grayscale': None}
        self.event_preprocessor = EventPreprocessor(options)
        self.intensity_rescaler = IntensityRescaler(options)
        self.image_filter = ImageFilter(options)
        self.unsharp_mask_filter = UnsharpMaskFilter(options, device=self.device)

    def update_reconstruction(self, event_tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            if isinstance(event_tensor, np.ndarray):
                event_tensor = torch.from_numpy(event_tensor)

            events = event_tensor.unsqueeze(dim=0)
            events = events.to(self.device)

            events = self.event_preprocessor(events)

            # Resize tensor to [1 x C x crop_size x crop_size] by applying zero padding
            events_for_each_channel = {'grayscale': self.crop.pad(events)}
            reconstructions_for_each_channel = {}
            # Reconstruct new intensity image for each channel (grayscale + RGBW if color reconstruction is enabled)
            for channel in events_for_each_channel.keys():
                new_predicted_frame, states = self.model(events_for_each_channel[channel],
                                                            self.last_states_for_each_channel[channel])

                if self.no_recurrent:
                    self.last_states_for_each_channel[channel] = None
                else:
                    self.last_states_for_each_channel[channel] = states

                # Output reconstructed image
                crop = self.crop if channel == 'grayscale' else self.crop_halfres  # type: ignore

                # Unsharp mask (on GPU)
                new_predicted_frame = self.unsharp_mask_filter(new_predicted_frame)
                # Intensity rescaler (on GPU)
                new_predicted_frame = self.intensity_rescaler(new_predicted_frame)
                reconstructions_for_each_channel[channel] = new_predicted_frame[0, 0, crop.iy0:crop.iy1,
                                                                                crop.ix0:crop.ix1].cpu().numpy()

                out = reconstructions_for_each_channel['grayscale']

            # Post-processing, e.g bilateral filter (on CPU)
            out = self.image_filter(out)
        return out  # type: ignore
