import argparse
from utils.misc import *

from train import train
from evaluate import evaluate


def main(args):
    if args.mode == "train":
        config = load_config_file(args.config_file_path)
        train(config)
    
    elif args.mode == "evaluate":
        evaluate(args)
    
    else:
        raise Warning(f"Unrecognized mode {args.mode}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(help="help for mode", dest="mode")
    
    train_parser: argparse.ArgumentParser = sub_parsers.add_parser("train", help="train model using a configuration file")
    train_parser.add_argument(
        "-c", "--config-file", help="path to config file", dest="config_file_path", type=str, required=False)
    
    evaluate_parser: argparse.ArgumentParser = sub_parsers.add_parser("evaluate", help="evaluate a model")
    evaluate_parser.add_argument("-d", "--dir", help="path to model run directory", dest="run_dir")
    
    args = parser.parse_args()
    main(args)