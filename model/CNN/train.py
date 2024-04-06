import argparse
import os
import sys

import training_loop

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import dataset_utils

def main():

    # arg parser
    parser = argparse.ArgumentParser(description="Train a CNN model for image colorization.")
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--ts', type=str, required=True)
    parser.add_argument('--interval', type=int, default=5)
    
    args = parser.parse_args()
    
    if args.ts == 'public':
        train_dataset, test_dataset = dataset_utils.make_public_dataloader(batch_size=args.batch, aug=args.aug)
    else:
        try:
            train_dataset, test_dataset = dataset_utils.make_custom_dataloader(path=args.ts, batch_size=args.batch, aug=args.aug)
        except FileNotFoundError:
            print('Invalid training set path')
            exit(1)

    training_loop.train(train_dataset, test_dataset, resume=args.resume, epochs=args.epochs, interval=args.interval)

if __name__ == '__main__':
    main()
