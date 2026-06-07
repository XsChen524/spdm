def add_manimamba_parser(parser):
    parser.add_argument(
        "--d_state",
        type=int,
        default=16,
        help="Temporal Mamba SSM state dimension",
    )
    parser.add_argument(
        "--expand",
        type=int,
        default=1,
        help="Temporal Mamba expansion factor",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help="SPD regularization term for covariance matrices",
    )
    parser.add_argument(
        "--cov_window",
        type=int,
        default=16,
        help="Sliding window size for covariance computation",
    )
    parser.add_argument(
        "--cov_stride",
        type=int,
        default=8,
        help="Sliding window stride for covariance computation",
    )
    parser.add_argument(
        "--cov_rank",
        type=int,
        default=0,
        help="Low-rank projection dimension for covariance (0=no projection)",
    )
    parser.add_argument(
        "--geo_d_model",
        type=int,
        default=64,
        help="Geometry Mamba model dimension",
    )
    parser.add_argument(
        "--geo_d_state",
        type=int,
        default=16,
        help="Geometry Mamba SSM state dimension",
    )
    parser.add_argument(
        "--geo_d_conv",
        type=int,
        default=4,
        help="Geometry Mamba local convolution width",
    )
    parser.add_argument(
        "--geo_expand",
        type=int,
        default=1,
        help="Geometry Mamba expansion factor",
    )
    parser.add_argument(
        "--geo_inject_threshold",
        type=int,
        default=100,
        help="Variable count threshold for adaptive injection: N<=threshold uses dt+lightB, N>threshold uses B+C",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        choices=[
            "tanh_alpha",
            "no_bc",
            "w_dt",
            "linear_interp",
            "geo_smooth_reg",
        ],
        help="Ablation variant for v4 experiments",
    )
