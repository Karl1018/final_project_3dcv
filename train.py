import argparse

import model.GAN as GAN
import model.diffusion as diffusion
import model.CNN as CNN
from utils import dataset_utils

def main():
    # TODO: help message

    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['GAN', 'diffusion', 'CNN'], help='Model to train.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training.')
    parser.add_argument('--ts', type=str, required=True, help='Training set to use (public, custom path.)')
    parser.add_argument('--aug', type=bool, default=False, help='Use data augmentation.')
    parser.add_argument('--batch', type=int, default=32, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--interval', type=int, default=5, help='Interval to save snapshot.')
    args = parser.parse_args()

    if args.ts == 'public':
        train_dataset, test_dataset = dataset_utils.make_public_dataloader(batch_size=args.batch, aug=args.aug)

    
    else:
        try:
            train_dataset, test_dataset = dataset_utils.make_custom_dataloader(path=args.ts, batch_size=args.batch, aug=args.aug)
        except FileNotFoundError:
            print('Invalid training set path')
            exit(1)
    
    if args.model == 'GAN':
        training_loop = GAN.training_loop
    elif args.model == 'diffusion':
        training_loop = diffusion.training_loop
    elif args.model == 'CNN':
        training_loop = CNN.training_loop

    training_loop.train(train_dataset, test_dataset, resume=args.resume, epochs=args.epochs, interval=args.interval)


if __name__ == '__main__':
    main()