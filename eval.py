import argparse
import torch
from network import VAE
from train_classifier import Net

def main():
    parser = argparse.ArgumentParser(description='Evaluation pipeline on MNIST')
    parser.add_argument('--NoiseType', type=str, nargs='+',
                        help='type of noise to apply to the evaluation (default: ["white","gaussian","saltepper"])')
    parser.add_argument('--PathVAE', type=str, help='Path to the VAE/iVAE model')
    parser.add_argument('--PathClassifier', type=str, help='Path to the classifier model')

    args = parser.parse_args()

    ## loading the VAE model
    vae_param = torch.load(args.PathVAE)['model']
    ## to do : automatically load the kwargs from the experimentation file
    kwargs = {'x_dim': 28**2, 'z_dim': 10, 'h_dim1': 512, 'h_dim2': 256}
    vae_model = VAE(**kwargs)
    vae_model.load_state_dict(vae_param)

    ## loading the classification model
    classif_param = torch.load(args.PathClassifier)
    classif_model = Net()
    classif_model.load_state_dict(classif_param)

    ## load the testing database


    ## evaluation loop
    for noise in args.NoiseType:
        print(noise)
        if noise == 'white':
            print('white')
        elif noise == 'saltpepper':
            print('to do')
        elif noise == 'gaussian':
            print('to do')


if __name__ == '__main__':
    main()