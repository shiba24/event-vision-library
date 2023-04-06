import argparse


def set_inference_options(parser: argparse.ArgumentParser) -> None:

    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.set_defaults(use_gpu=True)

    """ Post-processing / filtering """
    # (optional) unsharp mask
    parser.add_argument('--unsharp_mask_amount', default=0.3, type=float)
    parser.add_argument('--unsharp_mask_sigma', default=1.0, type=float)

    # (optional) bilateral filter
    parser.add_argument('--bilateral_filter_sigma', default=0.0, type=float)

    # (optional) flip the event tensors vertically
    parser.add_argument('--flip', dest='flip', action='store_true')
    parser.set_defaults(flip=False)

    """ Tone mapping (i.e. rescaling of the image intensities)"""
    parser.add_argument('--Imin', default=0.0, type=float,
                        help="Min intensity for intensity rescaling (linear tone mapping).")
    parser.add_argument('--Imax', default=1.0, type=float,
                        help="Max intensity value for intensity rescaling (linear tone mapping).")
    parser.add_argument('--auto_hdr', dest='auto_hdr', action='store_true',
                        help="If True, will compute Imin and Imax automatically.")
    parser.set_defaults(auto_hdr=False)
    parser.add_argument('--auto_hdr_median_filter_size', default=10, type=int,
                        help="Size of the median filter window used to smooth temporally Imin and Imax")

    """ Advanced parameters """
    # disable normalization of input event tensors (saves a bit of time, but may produce slightly worse results)
    parser.add_argument('--no-normalize', dest='no_normalize', action='store_true')
    parser.set_defaults(no_normalize=False)

    # disable recurrent connection (will severely degrade the results; for testing purposes only)
    parser.add_argument('--no-recurrent', dest='no_recurrent', action='store_true')
    parser.set_defaults(no_recurrent=False)
