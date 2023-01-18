import time
import argparse
import multiprocessing

import torch
from src.data.dataset import ButterflyDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

def timing(dataloader):
    res = []
    for _ in range(5):
        start = time.time()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx > args.batches_to_check:
                break
        end = time.time()

        res.append(end - start)

    res = np.array(res)
    print(f'Timing: {np.mean(res)}+-{np.std(res)}')
    return res

def errorbar(mean_list, std_list, worker_list):
    plt.errorbar(x=worker_list, y=mean_list, yerr=std_list, ecolor=['red', 'green', 'blue', 'cyan', 'magenta'])
    plt.xlabel("Number of workers")
    plt.ylabel("Mean processing time")
    plt.xticks(np.arange(len(mean_list)))
    plt.show()


if __name__ == '__main__':
    cores = multiprocessing.cpu_count()
    print(f"Number of cores: {cores}, Number of threads: {2 * cores}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_folder', default='', type=str) # "../../data/processed/train.pt"
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--create_errorbar', action='store_true')
    parser.add_argument('-batches_to_check', default=62, type=int)

    args = parser.parse_args()

    dataset = ButterflyDataset(args.path_to_folder)

    if args.create_errorbar:
        mean_list = []
        std_list = []
        worker_list = np.arange(args.num_workers)
        for i in worker_list:
            dataloader = DataLoader(dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=i,
                                    pin_memory=True)

            res = timing(dataloader)
            mean_list.append(np.mean(res))
            std_list.append(np.std(res))

        errorbar(mean_list, std_list, worker_list)


