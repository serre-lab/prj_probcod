import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from network import Classifier
import os

import tools



parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--cuda', default=True,
                    help='disables CUDA training')
parser.add_argument('--verbose', type=bool, default=True, help='show everything')
parser.add_argument('--print-freq', type=int, default=1, help='logging epoch interval')
parser.add_argument('--ckpt-freq', type=int, default=0, help='checkpoint epoch interval')
parser.add_argument('--eval-freq', type=int, default=20, help='evaluation epoch interval')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--zca_matrix', type=str, default='/cifs/data/tserre_lrs/projects/prj_predcoding/mnist_zca_matrix_2.npy', help='path to store the trained network')

parser.add_argument('--path', type=str, default='', help='path to store the trained network')

#parser.add_argument('--save-model', action='store_true', default=False,
 #                   help='For Saving the current Model')
#parser.add_argument('--device', type=str, default='cuda:7', help='gpu name')

# data
parser.add_argument('--data_dir', type=str, default='../DataSet/MNIST/', help='dataset path')

args = parser.parse_args()


def train(args, train_loader, model, optimizer):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        #if batch_idx % args.log_interval == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader), loss.item()))
        #if args.dry_run:
        #    break
    train_loss = train_loss/len(train_loader.dataset)
    mean_accuracy = 100. * correct / len(train_loader.dataset)
    train_results = {
        'train_NLL_loss': train_loss,
        'train_accuracy': mean_accuracy
    }
    return train_results


def evaluate(args, test_loader, model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    mean_accuracy = 100. * correct / len(test_loader.dataset)

    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #    test_loss, correct, len(test_loader.dataset),
    #    mean_accuracy))

    results = {
        'val_NLL_loss' : test_loss,
        'val_accuracy' : mean_accuracy
    }
    return results

def main(args):
    # Training settings

    torch.manual_seed(args.seed)

    kwargs = {
        'batch_size': args.batch_size,
        'num_workers': 8,
        'pin_memory': True,
        'shuffle': True
    }

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        # tools.WhitenMNIST(args.zca_matrix)
        ])

    dataset1 = datasets.MNIST(args.data_dir, train=True, download=False,
                       transform=transform)
    dataset2 = datasets.MNIST(args.data_dir, train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = Classifier().cuda()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train_results = train(args, train_loader, model, optimizer)

        if args.verbose:
            if epoch % args.print_freq == 0:
                print(' Train Epoch: {} -- Loss {:6.4f} -- accuracy {:6.2f}% '.format(
                    epoch,
                    train_results['train_NLL_loss'],
                    train_results['train_accuracy']
                ))

        if epoch % args.eval_freq == 0 and args.eval_freq != 0:

            val_results = evaluate(args, test_loader, model)

            print(' Eval Epoch: {} -- Loss {:6.4f} -- accuracy {:6.2f}% '.format(
                epoch,
                val_results['val_NLL_loss'],
                val_results['val_accuracy']
            ))

        scheduler.step()

    save_dict = {
        'model' : model.state_dict(),
        'config' : vars(args),
        'results' : train_results
    }
    if args.path != '':
        torch.save(save_dict,
                   os.path.join(args.path, 'model_results.pth')
                   )


if __name__ == '__main__':
    print(vars(args))
    ## prepare output directory
    os.makedirs(args.path)

    main(args)
