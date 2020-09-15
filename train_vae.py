import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from network import VAE, loss_function, enable_grad, disable_grad
from torchvision.utils import make_grid
import argparse
from utils import show
#matplotlib.rc('xtick', labelsize=15)
#import matplotlib
#matplotlib.rc('ytick', labelsize=15)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_svi', type=float, default = 1e-2, help='learning rate of the iterative inference')
    parser.add_argument('--nb_epoch', type=int, default=200, help='number of training epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of the amortized inference')
    parser.add_argument('--path', type=str, default='', help='path to store the trained network')
    parser.add_argument('--type', type=str, default='IVAE', help='could be IVAE or VAE')
    args = parser.parse_args()


    device = 'cuda:7'
    batch_size = 1024
    kwargs = {'batch_size': batch_size,
              'drop_last': True,
              'num_workers': 8}

    grid_param = {'padding': 2, 'normalize': True,
                  'pad_value': 1,
                  'nrow': 8}

    transform = transforms.Compose([
        transforms.ToTensor()])

    dataset1 = datasets.MNIST('../../DataSet/MNIST/', train=True, download=False, transform=transform)
    #dataset2 = datasets.MNIST('../../DataSet/MNIST/', train=False, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **kwargs, shuffle=True)
    #test_loader = torch.utils.data.DataLoader(dataset2, **kwargs, shuffle=True)

    torch.manual_seed(1)
    nb_epoch = 200
    lr = 1e-3
    z_dim = 10
    update_rate = 50
    h_dim1 = 256
    h_dim2 = 128
    nb_iteration = 50
    lr_SVI = args.lr_svi
    #lr_SVI = 1e-2
    #hybrid = True
    vae = VAE(x_dim = 28**2, z_dim=z_dim, h_dim1=h_dim1, h_dim2=h_dim2).to(device)
    mu_p = nn.Parameter()
    log_var_p = nn.Parameter()
    param_svi = [mu_p, log_var_p]
    # optimizer_SVI = torch.optim.SGD([{'params' : param_svi}], lr=lr_SVI, momentum=0.9)
    # optimizer_SVI = torch.optim.RMSprop([{'params' : param_svi}], lr=lr_SVI, momentum=0.9)
    optimizer_SVI = torch.optim.Adam([{'params': param_svi}], lr=args.lr_svi)

    for p in param_svi:
        p.requires_grad = False

    optimizer = torch.optim.Adam([{'params': vae.parameters(), 'lr': args.lr}])

    all_param = vae.param_enc + vae.param_dec + param_svi
    disable_grad(all_param)

    # for p in vae.parameters():
    #    print(p.requires_grad)

    # for p in param_svi:
    #    print(p.requires_grad)


    # scheduler = StepLR(optimizer, step_size=update_rate, gamma=0.5)
    for idx_epoch in range(args.nb_epoch):
        train_reco_loss = 0
        train_KL_loss = 0
        train_loss_gen = 0

        vae.train()

        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            data = torch.bernoulli(data)
            mu_p.data, log_var_p.data = vae.encoder(data.view(-1, 784))
            enable_grad(param_svi)

            for idx_it in range(nb_iteration):
                optimizer_SVI.zero_grad()
                z = vae.sampling(mu_p, log_var_p)
                reco = vae.decoder(z)
                loss_gen, reco_loss, KL_loss = loss_function(reco, data, mu_p, log_var_p)
                loss_gen.backward()
                optimizer_SVI.step()

            disable_grad(param_svi)

            enable_grad(vae.param_dec)

            optimizer.zero_grad()
            z = vae.sampling(mu_p, log_var_p)
            reco = vae.decoder(z)
            loss_gen, reco_loss, KL_loss = loss_function(reco, data, mu_p, log_var_p)

            loss_gen.backward()
            optimizer.step()

            disable_grad(vae.param_dec)

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

        if idx_epoch % 10 == 0:
            print("original image")
            img_to_plot = make_grid(data[0:8, :, :, :], **grid_param)
            show(img_to_plot.detach().cpu())
            print("recontruction image")
            x_reco = reco.view(len(data), 1, 28, 28)
            img_to_plot = make_grid(x_reco[0:8, :, :, :], **grid_param)
            show(img_to_plot.detach().cpu())

        # scheduler.step()

        if idx_epoch % update_rate == 0:
            print('\n\nNEW LEARNING RATElr={0:.1e}\n\n'.format(optimizer.param_groups[0]['lr']))

        print(' Train Epoch: {} -- Loss {:6.2f} (reco : {:6.2f} -- KL : {:6.2f}) '.format(
            idx_epoch,
            train_loss_gen / len(train_loader.dataset),
            train_reco_loss / len(train_loader.dataset),
            train_KL_loss / len(train_loader.dataset)
        ))

        if args.path != '':
            to_save =  {'model' : vae.state_dict(),
                        'type' : args.type}

            torch.save(to_save, args.path + '{}_lrsvi={}_lr={}.pth'.format(args.type, args.lr_svi, args.lr))


if __name__ == '__main__':
    main()



