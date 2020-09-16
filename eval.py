import argparse
import torch
from network import VAE, enable_grad, disable_grad, loss_function, Classifier
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from utils import show, GaussianSmoothing
import json

def white_noise(input, std):
    to_return = input + std * torch.randn_like(input)
    return to_return

def gaussian_noise(input, std):
    return input

def salt_pepper_noise(input, proba_ones):
    mask = torch.bernoulli(torch.ones_like(input)*proba_ones).bool()
    input[mask] = torch.bernoulli(torch.ones_like(input[mask])*0.5)
    return input

def main():
    parser = argparse.ArgumentParser(description='Evaluation pipeline on MNIST')
    parser.add_argument('--config', type=str, help='path of the .json configuration file that defines the \
                                                       the type of noise to use and their parameters')
    parser.add_argument('--PathVAE', type=str, help='Path to the VAE/iVAE model')
    parser.add_argument('--PathClassifier', type=str, help='Path to the classifier model')
    parser.add_argument('--batch_size', type=int, default='64' ,help='size of the testing set batch')
    parser.add_argument('--verbose', type=bool, default=False, help='monitor the evaluation process')

    args = parser.parse_args()

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
    vae_param = torch.load(args.PathVAE)['model'] ## to do : automatically load the kwargs from the experimentation file
    kwargs = {'x_dim': 28**2, 'z_dim': 10, 'h_dim1': 512, 'h_dim2': 256}
    vae_model = VAE(**kwargs)
    vae_model.load_state_dict(vae_param)
    mu_p = torch.nn.Parameter()
    log_var_p = torch.nn.Parameter()
    param_svi = [mu_p, log_var_p]
    lr_SVI = 1e-4
    optimizer_SVI = torch.optim.Adam([{'params' : param_svi}], lr=lr_SVI) ## to do : make the loading of the optimizer automatic
    disable_grad(param_svi)

    ## loading the classification model
    classif_param = torch.load(args.PathClassifier)
    classif_model = Classifier()
    classif_model.load_state_dict(classif_param)

    if torch.cuda.is_available():
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
    std = 0.5
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
                if args.verbose:
                    if batch_idx == 0:
                        img_to_plot = make_grid(data[0:8, :, :, :], **grid_param)
                        show(img_to_plot.detach().cpu(), title='original image',
                             saving_path='../simulation/Image/image_original_{}_{}.png'.format(noise_type,param_noise))

                data = noise_function[noise_type](data,param_noise)

                if args.verbose:
                    if batch_idx == 0:
                        img_to_plot = make_grid(data[0:8, :, :, :], **grid_param)
                        show(img_to_plot.detach().cpu(), title='Blurred image',
                             saving_path='../simulation/Image/image_blurred_{}_{}.png'.format(noise_type,param_noise))

                mu_p.data, log_var_p.data = vae_model.encoder(data.view(-1, 784))

                ## Iterative refinement of the posterior
                enable_grad(param_svi)

                for idx_it in range(1000):
                    optimizer_SVI.zero_grad()
                    z = vae_model.sampling(mu_p, log_var_p)
                    reco = vae_model.decoder(z)
                    loss_gen, reco_loss, KL_loss = loss_function(reco, data, mu_p, log_var_p)
                    loss_gen.backward()
                    optimizer_SVI.step()

                if args.verbose:
                    reco_to_plot = reco.view_as(data)
                    img_to_plot = make_grid(reco_to_plot[0:8, :, :, :], **grid_param)
                    show(img_to_plot.detach().cpu(), title='Reconstructed image',
                         saving_path='../simulation/Image/image_reconstructed_{}_{}.png'.format(noise_type, param_noise))
                disable_grad(param_svi)

                output = classif_model(reco.view_as(data))
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(label.view_as(pred)).sum().item()
                ##

                if batch_idx >= 0:
                    break

            if args.verbose:
                print(100. * correct / len(test_loader.dataset))
                #if batch_idx >=0:
                #    break

if __name__ == '__main__':
    main()