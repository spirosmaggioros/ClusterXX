import argparse
from .py_funcs import run_clustering

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
    clustering.set_defaults(func=run_clustering)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
