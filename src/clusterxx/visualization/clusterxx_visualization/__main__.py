import argparse
from .py_funcs import run_clustering, run_scatterplot

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="clusterxx",
        description="ClusterXX visualization options",
        usage="", #TODO
        add_help=False,
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    clustering = subparsers.add_parser("clustering", help="Select clustering mode")
    clustering.add_argument(
        "--filename",
        type=str,
        required=True,
        help="[REQUIRED] The filename where the input features and labels are stored",
    )

    clustering.add_argument(
        "--title",
        type=str,
        required=False,
        help="Set title on the plot",
    )

    clustering.add_argument(
        "--xlabel",
        type=str,
        default=None,
        required=False,
        help="Set xlabel on the plot",
    )

    clustering.add_argument(
        "--ylabel",
        type=str,
        default=None,
        required=False,
        help="Set ylabel on the plot",
    )

    clustering.add_argument(
        "--zlabel",
        type=str,
        default=None,
        required=False,
        help="Set zlabel on the plot(only for 3d plots)",
    )

    clustering.add_argument(
        "--labels",
        action="store_true",
        help="Use ground truth labels for decomposition/manifold plotting",
    )
    
    clustering.set_defaults(func=run_clustering)

    scatterplot = subparsers.add_parser("scatterplot", help="Select scatterplot mode(anything but clustering mode)")
    scatterplot.add_argument(
        "--filename",
        type=str,
        required=True,
        help="[REQUIRED] The filename where the input features and labels are stored",
    )

    scatterplot.add_argument(
        "--title",
        type=str,
        required=False,
        help="Set title on the plot",
    )

    scatterplot.add_argument(
        "--xlabel",
        type=str,
        default=None,
        required=False,
        help="Set xlabel on the plot",
    )

    scatterplot.add_argument(
        "--ylabel",
        type=str,
        default=None,
        required=False,
        help="Set ylabel on the plot",
    )

    scatterplot.add_argument(
        "--zlabel",
        type=str,
        default=None,
        required=False,
        help="Set zlabel on the plot(only for 3d plots)",
    )

    scatterplot.add_argument(
        "--labels",
        action="store_true",
        help="Only when the user pass the ground truth as well",
    )
    scatterplot.set_defaults(func=run_scatterplot)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
