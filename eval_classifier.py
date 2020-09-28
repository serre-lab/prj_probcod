import os

import json

import numpy as np
import pandas as pd

import torch

from torchvision import datasets, transforms

from network import Classifier

from tools import GaussianSmoothing

import time


import argparse


parser = argparse.ArgumentParser(description='Evaluation pipeline on MNIST')
## Path to the Classifier model, the classifier and the config file
parser.add_argument('--config', type=str, help='path of the .json configuration file that defines the \
                                                   the type of noise to use and their parameters')
parser.add_argument('--PathClassifier', type=str, help='Path to the classifier model')
parser.add_argument('--normalized_output', type=int, default=0, help='normalize the output of the blurring function between 0 and 1')


## Parameters of the evaluation process
parser.add_argument('--batch_size', type=int, default='64' ,help='size of the testing set batch')
parser.add_argument('--verbose', type=bool, default=False, help='monitor the evaluation process')
parser.add_argument('--cuda', type=bool, default=True, help='use GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--disp', type=bool, default=True, help='image display epoch interval')

### saving path
parser.add_argument('--path', type=str, default='', help='path to store the results of the evaluation')
parser.add_argument('--path_db', type=str, default='db_EVAL_original.csv', help='path to the training database')
parser.add_argument('--nb_class', type=int, default=10, help='number of class of the classifier')
parser.add_argument('--save_in_db', type=int, default=1, help='option to save the results in database')
parser.add_argument('--save_latent', type=int, default=0, help='option to save the statistics per sample')


args = parser.parse_args()


def white_noise(input, std):
    to_return = input + std * torch.randn_like(input)
    return to_return

def gaussian_noise(input, std):
    return input

def salt_pepper_noise(input, proba_ones):
    to_out = input.clone()
    mask = torch.bernoulli(torch.ones_like(input)*proba_ones).bool()
    to_out[mask] = torch.bernoulli(torch.ones_like(input[mask])*0.5)
    return to_out

def no_noise(input,param=None,idx_param=None):
    return input

def main(args):

    torch.manual_seed(args.seed)

    with open(args.config) as config_file:
        dico_config = json.load(config_file)


    noise_function = {'NoNoise' : no_noise,
                      'WhiteNoise': white_noise,
                      'Blurring': gaussian_noise,
                      'SaltPepper': salt_pepper_noise,
                      }

    ## setting the grid param
    grid_param = {'padding': 2, 'normalize': False,
                  'pad_value': 1,
                  'nrow': 8}


    ## loading the VAE model

    os.makedirs(args.path)

    ## loading the classification model
    pathClassifier_to_model = os.path.join(args.PathClassifier, "model_results.pth")

    cl_loading = torch.load(pathClassifier_to_model)
    args_cl = cl_loading['config']
    if "type" not in args_cl.keys():
        args_cl["type"] = 'CL'
    classif_model = Classifier()
    classif_model.load_state_dict(cl_loading['model'])

    if args.cuda:
        classif_model.cuda()

    ## load the testing database
    kwargs = {'batch_size': args.batch_size,
              'num_workers': 8,
              'pin_memory': True,
              'shuffle': False
            }

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    #dataset = datasets.MNIST('../DataSet/MNIST/', train=False,
    #                          transform=transform)
    dataset = datasets.QMNIST('../DataSet/QMNIST/', what='test', download=True,
                                                       transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset, **kwargs)


    df_results = pd.DataFrame(columns=['transform', 'param', 'accuracy'])

    ## evaluation loop
    for noise_type in dico_config.keys():
        for param_noise in dico_config[noise_type]['param']:

            correct_per_sample_to_save = torch.empty(0,).long()
            softmax_reshape_to_save = torch.empty(0, args.nb_class)
            prediction_to_save = torch.empty(0, ).long()
            label_to_save = torch.empty(0).long()

            if args.verbose:
                print('Evaluating on {} noise with parameter {}'.format(noise_type, param_noise))
            correct = 0

            if noise_type == 'Blurring':
                smoothing = GaussianSmoothing(1, 28, param_noise, normalize_output=args.normalized_output).cuda()
                noise_function['Blurring'] = smoothing.forward

            for batch_idx, (data, label) in enumerate(test_loader):
                data, label = data.cuda(), label.cuda()
                data_blurred = noise_function[noise_type](data,param_noise)

                if args.normalized_output == 1:
                    data_blurred = (data_blurred - 0.1307) / 0.3081

                output = classif_model(data_blurred)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                correct += pred.eq(label.view_as(pred)).sum().float()
                correct_per_sample = pred.eq(label.view_as(pred)) + 0


                if args.save_latent:
                    correct_per_sample_to_save = torch.cat(
                        [correct_per_sample_to_save, correct_per_sample.detach().cpu()], 0)
                    softmax_reshape_to_save = torch.cat(
                        [softmax_reshape_to_save, output.view(data.size(0), -1).detach().cpu()], 0)
                    prediction_to_save = torch.cat(
                        [prediction_to_save, pred.view(data.size(0)).detach().cpu()], 0)
                    label_to_save = torch.cat([label_to_save, label.detach().cpu()], 0)
            acc = 100.*(correct/len(dataset))

            print('Accuracy : {0}'.format(acc))

            best_accu = acc
            best_it = 0



            # put data here
            #results = {}
            filename = os.path.join(args.path, 'result_{}_{}.pth'.format(noise_type, param_noise))

            dico_result = {'accuracy': acc}

            if args.save_latent:
                dico_result['correct_per_sample'] = correct_per_sample_to_save.tolist()
                dico_result['softmax_per_sample'] = softmax_reshape_to_save.tolist()
                dico_result['prediction_per_sample'] = prediction_to_save.tolist()
                dico_result['labels'] = label_to_save.tolist()


            torch.save(dico_result, filename)
            if os.path.exists(args.path_db):
                df_results = pd.read_csv(args.path_db, index_col=0)
            else:
                df_results = pd.DataFrame()

            rows = {
                'eval_path':args.path,
                'model_path': args_cl['path'],
                'model_type': args_cl["type"],

                'batch_size': args.batch_size,

                'transform': noise_type,
                'param':param_noise,
                'normalize_output':args.normalized_output,

                'best_accu_eval': float('%.2f' % (best_accu)),
                'best_it_eval': int(best_it),
                'path_to_results': filename,
                'seed': args.seed
            }

            df_results = df_results.append(rows, ignore_index=True)

            if args.save_in_db == 1:
                print('save in db')
                df_results.to_csv(args.path_db)


if __name__ == '__main__':
    print(vars(args))
    ## prepare output directory


    main(args)