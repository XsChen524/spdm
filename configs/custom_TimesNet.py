def add_timesnet_parser(parser):
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="number of top periods for FFT in TimesBlock",
    )
    parser.add_argument(
        "--num_kernels",
        type=int,
        default=6,
        help="number of Conv2d kernels in Inception block",
    )
