import argparse
import logging

from dataloader import FraudDataset
from models import SimpleNet


def train(model, optimizer, train_data, lr, batch_size):
    # TODO: Implement training loop
    pass


def val(model, val_data, lr, batch_size):
    # TODO: Implement validation loop
    pass


def run_train_eval(model, optimizer, train_data, val_data, lr, epochs, batch_size):
    # TODO: Iterate between training and eval loops
    for e in epochs:
        train(model, optimizer, train_data, lr, batch_size)
        val(model, val_data, lr, batch_size)


def main():
    # Set up logging
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s(%(name)s): %(message)s')
    consH = logging.StreamHandler()
    consH.setFormatter(formatter)
    consH.setLevel(logging.DEBUG)
    log.addHandler(consH)

    # Parse command lines
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str,
                        help='Path to csv with data file.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01).')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs (default: 10).')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64).')
    parser.add_argument('--train-val-split', type=float, default=0.8,
                        help='Training vs. validation split (default: 0.8).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0).')
    opts = parser.parse_args()
    data_path = opts.data_path
    lr = opts.lr
    epochs = opts.epochs
    batch_size = opts.batch_size
    seed = opts.seed
    split_percent = opts.train_val_split
    log.info(f'Running fraud_net with the following parameters:\n- Learning Rate: {lr}\n- Epochs: {epochs}' +
             f'\n- Batch size: {batch_size}\n- Seed: {seed}' +
             f'\n- Training-validation split: {100*split_percent:.0f}-{100*(1-split_percent):.0f}')

    # Load data
    train_data = FraudDataset(csv_file=data_path, split='train', split_percent=split_percent, seed=seed)
    val_data = FraudDataset(csv_file=data_path, split='val', split_percent=split_percent, seed=seed)

    # Load model and optimizer
    # TODO: Create model script, import it, instantiate model here
    model = SimpleNet(in_dim=10, hidden_dim=100, out_dim=1)   # this is just an example
    optimizer = None
    # TODO: Instantiate optimizer

    # Run training-validation loops:
    run_train_eval(model, optimizer, train_data, val_data, lr, epochs, batch_size)


if __name__ == '__main__':
    main()
