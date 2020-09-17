import os

import json

import numpy as np
import pandas as pd

import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from network import VAE, iVAE, enable_grad, disable_grad, loss_function, Classifier
from utils import show, GaussianSmoothing

import argparse


parser = argparse.ArgumentParser(description='Evaluation pipeline on MNIST')
parser.add_argument('--config', type=str, help='path of the .json configuration file that defines the \
                                                   the type of noise to use and their parameters')
parser.add_argument('--PathVAE', type=str, help='Path to the VAE/iVAE model')
parser.add_argument('--PathClassifier', type=str, help='Path to the classifier model')
parser.add_argument('--batch_size', type=int, default='64' ,help='size of the testing set batch')
parser.add_argument('--verbose', type=bool, default=False, help='monitor the evaluation process')
parser.add_argument('--path', type=str, default='', help='path to store the results of the evaluation')
parser.add_argument('--cuda', type=bool, default=True, help='use GPUs')
parser.add_argument('--disp', type=bool, default=True, help='image display epoch interval')

args = parser.parse_args()


def white_noise(input, std):
    to_return = input + std * torch.randn_like(input)
    return to_return

def gaussian_noise(input, std):
    return input

def salt_pepper_noise(input, proba_ones):
    mask = torch.bernoulli(torch.ones_like(input)*proba_ones).bool()
    input[mask] = torch.bernoulli(torch.ones_like(input[mask])*0.5)
    return input

def main(args):

    with open(args.config) as config_file:
        dico_config = json.load(config_file)


    noise_function = {'white': white_noise,
                      'gaussian': gaussian_noise,
                      'saltpepper': salt_pepper_noise}

    ## setting the grid param
    grid_param = {'padding': 2, 'normalize': True,
                  'pad_value': 1,
                  'nrow': 8}


    ## loading the VAE model

    #vae_param = torch.load(args.PathVAE)['model'] ## to do : automatically load the kwargs from the experimentation file
    vae_loading = torch.load(args.PathVAE)
    args_vae = vae_loading['config']

    print('ARGS_VAE')
    print(args_vae)
    if args_vae['type'] == 'IVAE':
        vae_model = iVAE(x_dim=28**2, lr_svi=args_vae['lr_svi'], z_dim=args_vae['z_dim'], \
                         h_dim1=args_vae['arch'][0], h_dim2=args_vae['arch'][1],\
                         cuda=args.cuda)
        vae_model.load_state_dict(vae_loading['model'])
    elif args_vae['type'] == 'VAE':
        print('TO DO')
    elif args_vae['type'] == 'PC':
        print('TO DO')


    #kwargs = {'x_dim': 28**2, 'z_dim': 10, 'h_dim1': 512, 'h_dim2': 256}
    #vae_model = VAE(**kwargs)
    #vae_model.load_state_dict(vae_param)
    #mu_p = torch.nn.Parameter()
    #log_var_p = torch.nn.Parameter()
    #param_svi = [mu_p, log_var_p]
    #lr_SVI = 1e-4
    #optimizer_SVI = torch.optim.Adam([{'params' : param_svi}], lr=lr_SVI) ## to do : make the loading of the optimizer automatic
    #disable_grad(param_svi)

    ## loading the classification model
    cl_loading = torch.load(args.PathClassifier)
    args_cl = cl_loading['config']
    classif_model = Classifier()
    classif_model.load_state_dict(cl_loading['model'])

    if args.cuda:
        vae_model.cuda()
        classif_model.cuda()

    ## load the testing database
    kwargs = {'batch_size': args.batch_size,
              'num_workers': 8,
              'pin_memory': True,
              'shuffle': False
            }

    transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST('../../DataSet/MNIST/', train=False,
                              transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset, **kwargs)

    df_results = pd.DataFrame(columns=['transform', 'param', 'accuracy', 'out_file'])
    ## evaluation loop
    #for noise in args.NoiseType:
    for noise_type in dico_config.keys():
        for param_noise in dico_config[noise_type]:
            if args.verbose:
                print('Evaluating on {} noise with parameter {}'.format(noise_type, param_noise))
            correct = 0

            if noise_type == 'gaussian':
                smoothing = GaussianSmoothing(1, 28, param_noise).cuda()
                noise_function['gaussian'] = smoothing.forward

            for batch_idx, (data, label) in enumerate(test_loader):
                data, label = data.cuda(), label.cuda()
                data_blurred = noise_function[noise_type](data,param_noise)

                reco, z, mu_l_p, log_var_p, loss_gen, reco_loss, KL_loss = vae_model.forward_eval(data_blurred, nb_it=args_vae['nb_it'])
                output = classif_model(reco.view_as(data))
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                
                correct += pred.eq(label.view_as(pred)).sum().item()
            
            acc = correct/len(dataset)
            
            # put data here
            results = {}
            filename = os.path.join(args.path, 'result_{}_{}.npy'.format(noise_type, param_noise))
            np.save(filename, results)

            df_results.append({
                'transform': noise_type,
                'param': param_noise,
                'accuracy': acc,
                'out_file': filename,
            })
            
            # if args.disp:

            #     z = vae_model.sampling(phi.mu, phi.log_var)
            #     reco = vae_model.decoder(z)
            #     x_reco = reco.view_as(data)
            #     img_to_plot = make_grid(torch.cat([data[0:8, :, :, :], data_blurred[0:8, :, :, :], x_reco], 0), **grid_param)
            #     save_image(img_to_plot, fp=os.path.join(args.path, 'image_{}_{}.png'.format(noise_type, param_noise)))


            #     print(100. * correct/len(test_loader.dataset))

                ##

                #if batch_idx >= 0:
                #    break

            if args.verbose:
                print(100. * correct / len(test_loader.dataset))
                #if batch_idx >=0:
                #    break

    df_results.to_csv(os.path.join(args.path,'results.csv')) 
    # when reading use index_col=0

if __name__ == '__main__':
    print(vars(args))
    ## prepare output directory
    os.makedirs(args.path)

    main(args)