import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from tqdm import tqdm

from lib.dataset import MNISTDataset
from lib.networks import MLPClassifier, ConvClassifier


BATCH_SIZE = 8
NUM_WORKERS = 4


if __name__ == '__main__': 
    # Create the validation dataset and dataloader.
    valid_dataset = MNISTDataset(split='test')
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Create the network.
    #net = MLPClassifier()
    net = ConvClassifier()

    # Load best checkpoint.
    net.load_state_dict(torch.load(f'best-{net.codename}.pth')['net'])
    # Put the network in evaluation mode.
    net.eval()

    
    # Based on run_validation_epoch, write code for computing the 10x10 confusion matrix.
    confusion_matrix = np.zeros([10, 10])
    
    for batch in valid_dataloader:
        output = net(batch['input'])
        pred_dig = (torch.argmax(output,dim=1)).numpy()
        true_dig = batch['annotation'].numpy()
        confusion_matrix[pred_dig,true_dig]+=1
    
    # Plot the confusion_matrix.
    plt.figure(figsize=[5, 5])
    plt.imshow(confusion_matrix)
    plt.xticks(np.arange(10))
    plt.xlabel('prediction')
    plt.yticks(np.arange(10))
    plt.ylabel('annotation')
    for i in range(10):
        for j in range(10):
            plt.text(j, i, '%d' % (confusion_matrix[j, i]), ha='center', va='center', color='w', fontsize=12.5)
            # Note that because we were given that prediction
            #has to be plotted along x (see line 52) so the indices must
            # be flipped because rows represent predictions (as instructed).
            #To get the accurate plot we flipped the indices for this reason.
    plt.show()