
import os
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from network import VAE, PHI, loss_function, enable_grad, disable_grad
from torchvision.utils import make_grid, save_image
import argparse

import matplotlib as mpl
mpl.use('Agg')

from utils import show

#matplotlib.rc('xtick', labelsize=15)
#import matplotlib
#matplotlib.rc('ytick', labelsize=15)


parser = argparse.ArgumentParser()
    
# model
parser.add_argument('--type', type=str, default='IVAE', help='could be IVAE or VAE')
## To do : Do not hardcode the 3 layer architecture in the VAE network
parser.add_argument('--arch', type=int, nargs='+', default=[512,256,20], help='achitecture of the encoder')    
parser.add_argument('--nb_it', type=int, default=20, help='number of iteration if the iterative inference')
parser.add_argument('--lr_svi', type=float, default = 1e-2, help='learning rate of the iterative inference')

# training
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of the amortized inference')
parser.add_argument('--batch_size', type=int, default=1024, help='training batch size')
parser.add_argument('--nb_epoch', type=int, default=200, help='number of training epochs')

# other
parser.add_argument('--cuda', type=bool, default=True, help='use GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--verbose', type=bool, default=True, help='show everything')
parser.add_argument('--print-freq', type=int, default=1, help='logging iteration interval')
parser.add_argument('--disp-freq', type=int, default=10, help='image display iteration interval')
parser.add_argument('--ckpt-freq', type=int, default=20, help='checkpoint iteration interval')
parser.add_argument('--path', type=str, default='', help='path to store the trained network')

args = parser.parse_args()


def evaluate(args, test_loader, model):
    
    val_reco_loss = []
    val_KL_loss = []
    # train_loss_gen = []

    vae = model
    # vae, phi = model
    # param_svi= phi.parameters()
    
    # vae.eval()

    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.cuda()
        data = (data>0.001).float() #torch.bernoulli(data)
        
        batch_size = data.shape[0]

        phi = PHI()
        if args.cuda:
            phi = phi.cuda()
        param_svi = list(phi.parameters())                
        optimizer_SVI = torch.optim.Adam(phi.parameters(), lr=args.lr_svi)
        
        ## Initialise the posterior output
        with torch.no_grad():
            phi.mu_p.data, phi.log_var_p.data = vae.encoder(data.view(batch_size, -1))
        
        ## Iterative refinement of the posterior
        enable_grad(param_svi)
        for idx_it in range(args.nb_it-1):
            optimizer_SVI.zero_grad()
            z = vae.sampling(phi.mu_p, phi.log_var_p)
            reco = vae.decoder(z)
            loss_gen, reco_loss, KL_loss = loss_function(reco, data, phi.mu_p, phi.log_var_p, reduction='sum')
            loss_gen.backward()
            optimizer_SVI.step()

        optimizer_SVI.zero_grad()
        z = vae.sampling(phi.mu_p, phi.log_var_p)
        reco = vae.decoder(z)
        loss_gen, reco_loss, KL_loss = loss_function(reco, data, phi.mu_p, phi.log_var_p, reduction='none')
        
        disable_grad(param_svi)

        val_reco_loss += reco_loss.data.tolist()
        val_KL_loss += KL_loss.data.tolist()
        # train_loss_gen += loss_gen.data.tolist()

    results = { 
        'val_rec_loss_data': val_reco_loss, 
        'val_KL_div_data': val_KL_loss,
        'val_rec_loss': np.mean(val_reco_loss), 
        'val_KL_div': np.mean(val_KL_loss),
    }

    return results

def main(args):

    ## manual seed
    torch.manual_seed(args.seed)

    ## data 
    transform = transforms.Compose([transforms.ToTensor()])

    dataset1 = datasets.MNIST('../DataSet/MNIST/', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../DataSet/MNIST/', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, 
                                                batch_size=args.batch_size,
                                                drop_last=True,
                                                pin_memory=True,
                                                num_workers=8, 
                                                shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset2, 
                                                batch_size=args.batch_size,
                                                drop_last=False,
                                                pin_memory=True, 
                                                num_workers=8, 
                                                shuffle=False)
    
    ## model
    vae = VAE(x_dim = 28**2, h_dim1=args.arch[0], h_dim2=args.arch[1], z_dim=args.arch[-1])
    # phi = PHI()

    # optimizer_SVI = torch.optim.SGD([{'params' : param_svi}], lr=lr_SVI, momentum=0.9)
    # optimizer_SVI = torch.optim.RMSprop([{'params' : param_svi}], lr=lr_SVI, momentum=0.9)
    
    if args.cuda:
        vae = vae.cuda()
        # phi = phi.cuda()

    # param_svi = list(phi.parameters())
    
    optimizer = torch.optim.Adam(vae.parameters(), lr= args.lr)

    all_param = vae.param_enc + vae.param_dec # + param_svi
    disable_grad(all_param)
    # update_rate = 50
    # scheduler = StepLR(optimizer, step_size=update_rate, gamma=0.5)
    
    # display params
    grid_param = {'padding': 2, 'normalize': True,
                'pad_value': 1,
                'nrow': 8}

    ## training
    for idx_epoch in range(args.nb_epoch):
        train_reco_loss = 0
        train_KL_loss = 0
        train_loss_gen = 0

        vae.train()

        for batch_idx, (data, label) in enumerate(train_loader):
            
            if args.cuda:
                data = data.cuda()
            
            batch_size = data.shape[0]

            data = (data>0.001).float() 
            #data = torch.bernoulli(data)
            
            ## Initialise the posterior output
            phi = PHI()
            if args.cuda:
                phi = phi.cuda()
            param_svi = list(phi.parameters())                
            optimizer_SVI = torch.optim.Adam(phi.parameters(), lr=args.lr_svi)
            
            phi.mu_p.data, phi.log_var_p.data = vae.encoder(data.view(batch_size, -1))

            ## Iterative refinement of the posterior
            enable_grad(param_svi)
            for idx_it in range(args.nb_it):
                optimizer_SVI.zero_grad()
                z = vae.sampling(phi.mu_p, phi.log_var_p)
                reco = vae.decoder(z)
                loss_gen, reco_loss, KL_loss = loss_function(reco, data, phi.mu_p, phi.log_var_p, reduction='sum')
                loss_gen.backward()
                optimizer_SVI.step()
            disable_grad(param_svi)

            ## Amortized learning of the likelihood parameter
            enable_grad(vae.param_dec)
            optimizer.zero_grad()
            z = vae.sampling(phi.mu_p, phi.log_var_p)
            reco = vae.decoder(z)
            loss_gen, reco_loss, KL_loss = loss_function(reco, data, phi.mu_p, phi.log_var_p)
            loss_gen.backward()
            optimizer.step()
            disable_grad(vae.param_dec)

            ## Amortized learning of the posterior parameter
            enable_grad(vae.param_enc)
            optimizer.zero_grad()
            mu, log_var = vae.encoder(data.view(-1, 784))
            z = vae.sampling(mu, log_var)
            reco = vae.decoder(z)
            loss_gen, reco_loss, KL_loss = loss_function(reco, data, mu, log_var)
            loss_gen.backward()
            optimizer.step()
            disable_grad(vae.param_enc)


            train_reco_loss += reco_loss
            train_KL_loss += KL_loss
            train_loss_gen += loss_gen

        # scheduler.step()
        
        if args.verbose:
            # if idx_epoch % args.print_freq == 0:
            #     print('\n\nNEW LEARNING RATElr={0:.1e}\n\n'.format(optimizer.param_groups[0]['lr']))
        
            if idx_epoch % args.print_freq == 0:
                print(' Train Epoch: {} -- Loss {:6.5f} (reco : {:6.5f} -- KL : {:6.5f}) '.format(
                    idx_epoch,
                    train_loss_gen / len(train_loader.dataset),
                    train_reco_loss / len(train_loader.dataset),
                    train_KL_loss / len(train_loader.dataset)
                ))
            
            if idx_epoch % args.disp_freq == 0:
                # print("original image")
                # img_to_plot = make_grid(data[0:8, :, :, :], **grid_param)
                # show(img_to_plot.detach().cpu())
                # print("recontructed image")
                # x_reco = reco.view(len(data), 1, 28, 28)
                # img_to_plot = make_grid(x_reco[0:8, :, :, :], **grid_param)
                # show(img_to_plot.detach().cpu())

                x_reco = reco.view(len(data), 1, 28, 28)
                img_to_plot = make_grid(torch.cat([data[0:8, :, :, :], x_reco[0:8, :, :, :]], 0), **grid_param)
                save_image(img_to_plot, fp=os.path.join(args.path, 'recs_epoch{}.png'.format(idx_epoch)))

        if args.path != '' and idx_epoch % args.ckpt_freq==0:
            save_dict =  {
                'model' : vae.state_dict(),
                'optim': optimizer_SVI.state_dict(),
                'config': vars(args),
            }

            torch.save(save_dict, 
                        os.path.join(args.path, 'model_{}.pth'.format(idx_epoch))
                        )

    avg_train_rec_loss = train_reco_loss / len(train_loader.dataset)
    avg_train_KL_loss = train_KL_loss / len(train_loader.dataset)
    
    results={
        'train_rec_loss': avg_train_rec_loss, 
        'train_KL_div': avg_train_KL_loss,
    }
    
    val_results = evaluate(args, test_loader, vae)
    
    results.update(val_results)
    save_dict =  {
        'model' : vae.state_dict(),
        'optim': optimizer_SVI.state_dict(),
        'config': vars(args),
        'results': results,
    }

    torch.save(save_dict, 
                os.path.join(args.path, 'model_results.pth')
                )

if __name__ == '__main__':
    
    print(vars(args))
    ## prepare output directory
    os.makedirs(args.path)

    main(args)



