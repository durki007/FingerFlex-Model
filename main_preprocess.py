import os
import sys
import argparse
from src.data_preprocessing import PreprocessingPipeline


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess data')

    parser.add_argument(
        '--save_dir', type=str,
        default=PreprocessingPipeline.DEFAULT_PREPROCESSED_DATA_DIR,
        help='Directory to save processed data'
    )
    parser.add_argument(
        '--raw_data_dir', type=str,
        default=PreprocessingPipeline.DEFAULT_RAW_DATA_DIR,
        help='Directory to raw data'
    )
    parser.add_argument(
        '--time_delay_secs', type=int,
        default=PreprocessingPipeline.DEFAULT_TIME_DELAY_SECS,
        help='Time delay in seconds'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = PreprocessingPipeline(
        save_dir=args.save_dir,
        raw_data_dir=args.raw_data_dir,
        time_delay_secs=args.time_delay_secs,
        log_func=print,
    )
    pipeline.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())


