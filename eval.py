import os

import json

import numpy as np
import pandas as pd

import torch

from torchvision import datasets, transforms

from torchvision.utils import make_grid, save_image

from network import VAE, iVAE, enable_grad, disable_grad, loss_function, Classifier

from tools import GaussianSmoothing

import time


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
parser.add_argument('--nb_it_eval', type=int, default='0' ,help='iteration of the inference process in the evaluation \
                                                                procedure')
parser.add_argument('--freq_extra', type=int, default='0' ,help='frequence of evaluation of the inference process')
parser.add_argument('--lr_svi_eval', type=float, default='0' ,help='learning_rate of the inference during the \
                                                                    evalutaion process')

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

def main(args):

    with open(args.config) as config_file:
        dico_config = json.load(config_file)


    noise_function = {'white': white_noise,
                      'gaussian': gaussian_noise,
                      'saltpepper': salt_pepper_noise}

    ## setting the grid param
    grid_param = {'padding': 2, 'normalize': False,
                  'pad_value': 1,
                  'nrow': 8}


    ## loading the VAE model

    #vae_param = torch.load(args.PathVAE)['model'] ## to do : automatically load the kwargs from the experimentation file

    vae_loading = torch.load(args.PathVAE)
    args_vae = vae_loading['config']

    if args_vae['type'] == 'IVAE':
        if args.lr_svi_eval !=0 :
            lr_svi = args.lr_svi_eval
        else :
            lr_svi = args_vae['lr_svi']
        vae_model = iVAE(x_dim=28**2, lr_svi=lr_svi, z_dim=args_vae['z_dim'], \
                         h_dim1=args_vae['arch'][0], h_dim2=args_vae['arch'][1],\
                         cuda=args.cuda)
        vae_model.load_state_dict(vae_loading['model'])
        if args.nb_it_eval == 0:
            args.nb_it_eval = args_vae['nb_it']
    elif args_vae['type'] == 'VAE':
        vae_model = VAE(x_dim=28 ** 2, z_dim=args_vae['z_dim'], \
                         h_dim1=args_vae['arch'][0], h_dim2=args_vae['arch'][1])
        vae_model.load_state_dict(vae_loading['model'])
    elif args_vae['type'] == 'PC':
        print('TO DO')

    disable_grad(vae_model.parameters())

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
              'num_workers': 1,
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

    #df_results = pd.DataFrame(columns=['transform', 'param', 'accuracy', 'out_file'])
    df_results = pd.DataFrame(columns=['transform', 'param', 'accuracy'])

    ## evaluation loop
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
                if args_vae['type'] == 'IVAE':
                    reco, z, mu_l_p, log_var_p, loss_gen, reco_loss, KL_loss = vae_model.forward_eval(data_blurred, nb_it=args.nb_it_eval, freq_extra=args.freq_extra)
                elif args_vae['type'] == 'VAE':
                    reco, z, mu_l_p, log_var_p, loss_gen, reco_loss, KL_loss = vae_model.forward_eval(data_blurred)

                if args.freq_extra != 0:
                    reco_size = reco.size()
                    reco = reco.view(-1,1, 28, 28)
                    label = label.unsqueeze(1).repeat(1,reco_size[1]).view(-1)
                else :
                    reco = reco.view_as(data)

                output = classif_model(reco)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                if args.freq_extra !=0:
                    correct += pred.eq(label.view_as(pred)).view(reco_size[0],reco_size[1]).sum(dim=0)

                else :
                    correct += pred.eq(label.view_as(pred)).sum().item()

            acc = correct.float()/len(dataset)



            print(acc)
            # put data here
            #results = {}
            #filename = os.path.join(args.path, 'result_{}_{}.npy'.format(noise_type, param_noise))
            #np.save(filename, results)

            df_results = df_results.append({
                'transform': noise_type,
                'param': param_noise,
                'accuracy': acc
                #'out_file': filename,
            }, ignore_index=True)
            
            if args.disp:
                #x_reco = reco.view_as(data)


                to_plot = torch.cat([data[0:8, :, :, :], data_blurred[0:8, :, :, :]],0)
                if args.freq_extra != 0:
                    reco = reco.view(reco_size[0],reco_size[1],reco.size(-3),reco.size(-2), reco.size(-1))
                    reco = reco.transpose(0,1)
                    reco = reco[:,0:8,:,:,:].reshape(-1,reco.size(-3),reco.size(-2), reco.size(-1))
                    to_plot = torch.cat([to_plot,reco],0)

                else :
                    to_plot = torch.cat([to_plot,reco[0:8,:,:,:]],0)


                img_to_plot = make_grid(to_plot, **grid_param)
                #img_to_plot = make_grid(torch.cat([data[0:8, :, :, :], data_blurred[0:8, :, :, :], x_reco[0:8]], 0), **grid_param)

                save_image(img_to_plot, fp=os.path.join(args.path, 'image_{}_{}.png'.format(noise_type, param_noise)))


                ##

                #if batch_idx >= 0:
                #    break

            #if args.verbose:
            #    print(100. * correct / len(test_loader.dataset))
                #if batch_idx >=0:
                #    break

    df_results.to_csv(os.path.join(args.path,'results.csv')) 
    # when reading use index_col=0

if __name__ == '__main__':
    print(vars(args))
    ## prepare output directory
    os.makedirs(args.path)

    main(args)