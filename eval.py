import os

import json

import numpy as np
import pandas as pd

import torch

from torchvision import datasets, transforms

from torchvision.utils import make_grid, save_image

from network import VAE, iVAE, PCN, IAI, enable_grad, disable_grad, loss_function, loss_function_pc, Classifier

from tools import GaussianSmoothing, normalize_data

import time


import argparse


parser = argparse.ArgumentParser(description='Evaluation pipeline on MNIST')
## Path to the VAE model, the classifier and the config file
parser.add_argument('--config', type=str, help='path of the .json configuration file that defines the \
                                                   the type of noise to use and their parameters')
parser.add_argument('--PathVAE', type=str, help='Path to the VAE/iVAE model')
parser.add_argument('--PathClassifier', type=str, help='Path to the classifier model')
parser.add_argument('--normalized_output', type=int, default=0, help='normalize the output of the blurring function between 0 and 1')


## Parameters of the evaluation process
parser.add_argument('--batch_size', type=int, default='64' ,help='size of the testing set batch')
parser.add_argument('--verbose', type=bool, default=False, help='monitor the evaluation process')
parser.add_argument('--freq_extra', type=int, default='0' ,help='frequence of evaluation of the inference process')
parser.add_argument('--cuda', type=bool, default=True, help='use GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--disp', type=bool, default=True, help='image display epoch interval')

## parameter of the VAE model during evaluation
parser.add_argument('--nb_it_eval', type=int, default='0' ,help='iteration of the inference process in the evaluation \
                                                                procedure')
parser.add_argument('--svi_lr_eval', type=float, default='0' ,help='learning_rate of the inference during the \
                                                                    evalutaion process')
parser.add_argument('--svi_optimizer_eval', type=str, default = 'Adam', help='type of the inference optimizer')
## saving path
parser.add_argument('--path', type=str, default='', help='path to store the results of the evaluation')
parser.add_argument('--path_db', type=str, default='db_EVAL.csv', help='path to the training database')
parser.add_argument('--save_in_db', type=int, default=1, help='1 to save in the database, 0 otherwise')
parser.add_argument('--save_latent', type=int, default=0, help='1 to save the latent space of the first batch, 0 otherwise')
parser.add_argument('--denoising_baseline', type=int, default=0, help='Compute the ELBO for a denoising framework')





args = parser.parse_args()


def main(args):

    torch.manual_seed(args.seed)
    with open(args.config) as config_file:
        dico_config = json.load(config_file)

    def white_noise(input, std):
        input_filtered = input + std * torch.randn_like(input)
        return input_filtered

    def gaussian_noise(input, std=None, idx_param=None):
        return input

    def salt_pepper_noise(input, proba_ones, idx_param=None):
        to_out = input.clone()
        mask = torch.bernoulli(torch.ones_like(input) * proba_ones).bool()
        to_out[mask] = torch.bernoulli(torch.ones_like(input[mask]) * 0.5)
        return to_out

    def no_noise(input,param=None,idx_param=None):
        return input



    noise_function = {'NoNoise' : no_noise,
                      'WhiteNoise': white_noise,
                      'Blurring': gaussian_noise,
                      'SaltPepper': salt_pepper_noise}

    ## setting the grid param
    grid_param = {'padding': 2, 'normalize': True,
                  'pad_value': 1,
                  'nrow': 8}


    ## loading the VAE model
    pathVAE_to_model = os.path.join(args.PathVAE, "model_results.pth")
    vae_loading = torch.load(pathVAE_to_model)
    args_vae = vae_loading['config']

    if 'beta' not in args_vae.keys():
        args_vae['beta']=1

    args.path = args.path + "_{}_[{},{},{}]_beta={}".format(args_vae['type'],
                                                    args_vae['arch'][0],args_vae['arch'][1],args_vae['z_dim'],args_vae['beta'])
    
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    if args_vae['type'] == 'IVAE':
        if (args.svi_lr_eval !=  args_vae['svi_lr']) and (args.svi_lr_eval !=0):
            print('svi_lr training : {} -- svi_lr eval :{}'.format(args_vae['svi_lr'], args.svi_lr_eval))
        if args.nb_it_eval != args_vae['nb_it'] and  (args.nb_it_eval !=0):
            print('svi_nb_it training : {} -- svi_nb_it eval :{}'.format(args_vae['nb_it'] , args.nb_it_eval))
        if args.svi_optimizer_eval != args_vae['svi_optimizer']:
            print('svi_optimizer training : {} -- svi_optimizer eval :{}'.format(args_vae['svi_optimizer'], args.svi_optimizer_eval))

        vae_model = iVAE(x_dim=28**2, lr_svi=args.svi_lr_eval, z_dim=args_vae['z_dim'],
                         h_dim1=args_vae['arch'][0], h_dim2=args_vae['arch'][1],
                         activation=args_vae['activation_function'], svi_optimizer=args.svi_optimizer_eval,
                         cuda=args.cuda, beta=args_vae['beta'], decoder_type=args_vae['decoder_type'])
        vae_model.load_state_dict(vae_loading['model'])

    elif args_vae['type'] == 'IAI':
        vae_model = IAI(x_dim=28**2, z_dim=args_vae['z_dim'],
                         h_dim1=args_vae['arch'][0], h_dim2=args_vae['arch'][1],
                         activation=args_vae['activation_function'], 
                         beta=args_vae['beta'], decoder_type=args_vae['decoder_type'])
        vae_model.load_state_dict(vae_loading['model'])

    elif args_vae['type'] == 'VAE':
        vae_model = VAE(x_dim=28 ** 2, z_dim=args_vae['z_dim'], \
                         h_dim1=args_vae['arch'][0], h_dim2=args_vae['arch'][1],
                        activation=args_vae['activation_function'], beta=args_vae['beta'], decoder_type=args_vae['decoder_type'])
        vae_model.load_state_dict(vae_loading['model'])
    
    elif args_vae['type'] == 'PCN':
        if (args.svi_lr_eval !=  args_vae['svi_lr']) and (args.svi_lr_eval !=0):
            print('svi_lr training : {} -- svi_lr eval :{}'.format(args_vae['svi_lr'], args.svi_lr_eval))
        if args.nb_it_eval != args_vae['nb_it'] and  (args.nb_it_eval !=0):
            print('svi_nb_it training : {} -- svi_nb_it eval :{}'.format(args_vae['nb_it'] , args.nb_it_eval))
        if args.svi_optimizer_eval != args_vae['svi_optimizer']:
            print('svi_optimizer training : {} -- svi_optimizer eval :{}'.format(args_vae['svi_optimizer'], args.svi_optimizer_eval))

        vae_model = PCN(x_dim=28**2, lr_svi=args.svi_lr_eval, z_dim=args_vae['z_dim'],
                         h_dim1=args_vae['arch'][0], h_dim2=args_vae['arch'][1],
                         activation=args_vae['activation_function'], svi_optimizer=args.svi_optimizer_eval,
                         cuda=args.cuda, beta=args_vae['beta'], decoder_type=args_vae['decoder_type'])
        vae_model.load_state_dict(vae_loading['model'])

    # disable_grad(vae_model.parameters())


    ## loading the classification model
    pathClassifier_to_model = os.path.join(args.PathClassifier, "model_results.pth")

    cl_loading = torch.load(pathClassifier_to_model)
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

    if args_vae['decoder_type'] == 'bernoulli':
        transform = transforms.Compose([
            transforms.ToTensor()])

    elif args_vae['decoder_type'] == 'gaussian':
        transform = transforms.Compose([
            transforms.ToTensor()])




    dataset = datasets.MNIST('../DataSet/MNIST/', train=False,
                              transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset, **kwargs)

    

    ## evaluation loop
    for noise_type in dico_config.keys():


        for idx_param, param_noise in enumerate(dico_config[noise_type]['param']):

            if 'lr_svi' in dico_config[noise_type].keys():
                vae_model.lr_svi = dico_config[noise_type]['lr_svi']

            if args.verbose:
                print('Evaluating on {} noise with parameter {}'.format(noise_type, param_noise))
            correct = 0

            if noise_type == 'Blurring':
                smoothing = GaussianSmoothing(1, 10, param_noise, normalize_output=args.normalized_output).cuda()
                noise_function['Blurring'] = smoothing.forward

            for batch_idx, (data, label) in enumerate(test_loader):
                data, label = data.cuda(), label.cuda()
                data_blurred = noise_function[noise_type](data,param_noise)


                if args.normalized_output == 1:
                    data = (data - 0.1307)/0.3081
                    data_blurred = (data_blurred - 0.1307)/0.3081

                if args.denoising_baseline == 1:
                    reco, z, mu_l_p, log_var_p, loss_gen, reco_loss, KL_loss, nb_it_l = vae_model.forward_eval(
                        data_blurred, x_clear=data,
                        nb_it=args.nb_it_eval, freq_extra=args.freq_extra)
                else :
                    reco, z, mu_l_p, log_var_p, loss_gen, reco_loss, KL_loss, nb_it_l = vae_model.forward_eval(data_blurred,
                                                                                                           nb_it=args.nb_it_eval, freq_extra=args.freq_extra)
                if batch_idx == 1 and args.save_latent:
                    mu_l_p_to_save = mu_l_p
                    log_var_p_to_save = log_var_p


                if (args.freq_extra != 0) and (args_vae['type'] in ['IVAE','PCN', 'IAI']):
                    reco_size = reco.size()
                    reco = reco.view(-1,1, 28, 28)
                    label = label.unsqueeze(1).repeat(1,reco_size[1]).view(-1)
                else :
                    reco = reco.view_as(data)

                output = classif_model(reco)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                if (args.freq_extra !=0) and args_vae['type'] in ['IVAE','PCN', 'IAI']:
                    correct += pred.eq(label.view_as(pred)).view(reco_size[0],reco_size[1]).sum(dim=0).float()

                else :
                    correct += pred.eq(label.view_as(pred)).sum().float()

            acc = 100.*(correct/len(dataset))

            #print('Accuracy : {0}'.format(acc))
            if args_vae['type'] in ['IVAE','PCN', 'IAI']:
                accu = acc.cpu()

                best_accu, indices = accu.max(0)
                best_it = int(nb_it_l[indices])
                best_accu = float(best_accu)
            else :
                best_accu = acc
                best_it = 0


            filename = os.path.join(args.path, 'result_{}_{}.pth'.format(noise_type, param_noise))
            dico_result = {'accuracy': acc.tolist()}

            if args_vae['type'] in ['IVAE','PCN']:
                dico_result = {'accuracy' : acc.tolist()}
            elif args_vae['type'] == 'VAE':
                dico_result = {'accuracy': acc.tolist()}
            print('accuracy', dico_result['accuracy'])

            if args.save_latent:
                dico_result['latent_mu'] = mu_l_p_to_save.tolist()
                dico_result['latent_log_var'] = log_var_p_to_save.tolist()
                #print(dico_result['latent_mu'])

            torch.save(dico_result, filename)

            if args.disp:
                to_plot = torch.cat([data[0:8, :, :, :], data_blurred[0:8, :, :, :]],0)
                if (args.freq_extra != 0) and (args_vae['type'] in ['IVAE','PCN', 'IAI']):
                    reco = reco.view(reco_size[0],reco_size[1],reco.size(-3),reco.size(-2), reco.size(-1))
                    reco = reco.transpose(0,1)
                    reco = reco[:,0:8,:,:,:].reshape(-1,reco.size(-3),reco.size(-2), reco.size(-1))
                    to_plot = torch.cat([to_plot,reco],0)

                else :
                    to_plot = torch.cat([to_plot,reco[0:8,:,:,:]],0)
                img_to_plot = make_grid(to_plot, **grid_param)
                save_image(img_to_plot, fp=os.path.join(args.path, 'image_{}_{}.png'.format(noise_type, param_noise)))

            
            if not os.path.isfile(args.path_db):
                df_results = pd.DataFrame()
            else:
                df_results = pd.read_csv(args.path_db, index_col=0)

            rows = {
                'eval_path':args.path,
                'model_path': args_vae['path'],
                'model_type': args_vae["type"],
                'model_architecture': args_vae["arch"],
                'model_z_dim': args_vae["z_dim"],
                'beta' : args_vae['beta'],

                'svi_optimizer_training': args_vae["svi_optimizer"],
                'svi_lr_training': args_vae["svi_lr"],
                'svi_nb_it_training': args_vae["nb_it"],


                'svi_optimizer_eval': args.svi_optimizer_eval,
                'svi_lr_eval': vae_model.lr_svi,
                'svi_nb_it_eval': args.nb_it_eval,
                'batch_size': args.batch_size,

                'transform': noise_type,
                'param':param_noise,
                'normalize_output':args.normalized_output,

                'best_accu_eval': float('%.2f' % (best_accu)),
                'best_it_eval': int(best_it),
                'denoising_baseline':args.denoising_baseline,
                'path_to_results': filename,
                'seed': args.seed
            }

            df_results = df_results.append(rows, ignore_index=True)

            if args.save_in_db == 1:
                df_results.to_csv(args.path_db)

if __name__ == '__main__':
    print(vars(args))
    ## prepare output directory


    main(args)
