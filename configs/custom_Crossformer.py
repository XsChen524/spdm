def add_crossformer_parser(parser):
    parser.add_argument(
        "--seg_len",
        type=int,
        default=6,
        help="segment length for DSW embedding in Crossformer",
    )
    parser.add_argument(
        "--win_size",
        type=int,
        default=2,
        help="window size for segment merging in Crossformer",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="use mean of past series as baseline for Crossformer prediction",
    )
