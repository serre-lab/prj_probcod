import torch.nn as nn
import torch
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class View(nn.Module):
    def __init__(self, size=(1,28,28)):
        super(View, self).__init__()
        self.size = size
    def forward(self, input):
        return input.view(input.size(0), self.size[0], self.size[1], self.size[2])


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size[0], size[1])

class VAE(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim1=512, h_dim2=256, activation='tanh', layer='fc', beta=1, decoder_type='Bernoulli'):
        super(VAE, self).__init__()
        self.layer = layer
        self.beta = beta
        self.decoder_type=decoder_type

        if activation == 'tanh':
            self.activ_func = torch.nn.Tanh()
        elif activation == 'relu':
            self.activ_func = torch.nn.ReLU()

        if layer == 'fc':
            # encoder part
            self.ff1 = nn.Linear(x_dim, h_dim1)
            self.ff2 = nn.Linear(h_dim1, h_dim2)

            # decoder part
            self.fb4 = nn.Linear(z_dim, h_dim2)
            self.fb5 = nn.Linear(h_dim2, h_dim1)
            self.fb6 = nn.Linear(h_dim1, x_dim)



            self.enc = nn.Sequential(self.ff1, self.activ_func, self.ff2, self.activ_func)
            self.dec = nn.Sequential(self.fb4, self.activ_func, self.fb5, self.activ_func, self.fb6)

        elif layer == 'conv':
            # encoder part
            self.reshape1 = View(size=(1,28,28))
            self.ff1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
            self.pool1 = nn.MaxPool2d(2)
            self.ff2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
            self.pool2 = nn.MaxPool2d(2)
            h_dim2 = 800

            self.fb4 = nn.Linear(z_dim, h_dim2)
            self.reshape2 = View(size=(32,5,5))
            self.unpool2 = nn.Upsample(scale_factor = 2)
            self.fb5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1)
            self.unpool1 = nn.Upsample(scale_factor = 2)
            self.fb6 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1)
            self.unpool3 = nn.Upsample(size=(28,28))


            self.enc = nn.Sequential(self.reshape1, self.ff1, self.activ_func, self.pool1, self.ff2, self.activ_func, self.pool2, Flatten())
            self.dec = nn.Sequential(self.fb4, self.activ_func,self.reshape2,self.unpool2 ,self.fb5, self.activ_func, self.unpool1, self.fb6, self.unpool3)


            # decoder part


        self.ff3_mu = nn.Linear(h_dim2, z_dim)
        self.ff3_var = nn.Linear(h_dim2, z_dim)



        #self.enc = nn.Sequential(self.ff1, self.activ_func, self.ff2, self.activ_func)
        #self.dec = nn.Sequential(self.fb4, self.activ_func, self.fb5, self.activ_func)

        # self.fb6_mu = nn.Linear(h_dim1, x_dim)
        # self.fb6_var = nn.Linear(h_dim1, x_dim)

        self.param_enc = [self.ff1.weight, self.ff1.bias, self.ff2.weight, self.ff2.bias,
                          self.ff3_mu.weight, self.ff3_mu.bias, self.ff3_var.weight, self.ff3_var.bias]

        self.param_dec = [self.fb4.weight, self.fb4.bias, self.fb5.weight, self.fb5.bias,
                          self.fb6.weight, self.fb6.bias]

        self.z_dim = z_dim

    def encoder(self, x):

        #h = torch.tanh(self.ff1(x))
        #h = torch.tanh(self.ff2(h))
        h = self.enc(x)
        mu = self.ff3_mu(h)
        log_var = self.ff3_var(h)
        return mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def decoder(self, z):

        h  = self.dec(z)

        #if self.decoder_type=='bernoulli':

        if self.decoder_type == 'bernoulli':
            to_return = torch.sigmoid(h)
        elif self.decoder_type == 'gaussian':
            to_return = h

        return to_return




    def step(self, data):
        mu, log_var = self.encoder(torch.flatten(data, start_dim=1))
        z = self.sampling(mu, log_var)
        reco = self.decoder(z)
        loss_gen, reco_loss, KL_loss = loss_function(reco, data, mu, log_var,
                                                     reduction='sum', beta=self.beta, decoder_type=self.decoder_type)
        loss_gen.backward()

        return reco, z, loss_gen, reco_loss, KL_loss

    def forward_eval(self, data, reduction='sum', nb_it=None, freq_extra=None):
        mu, log_var = self.encoder(torch.flatten(data, start_dim=1))
        z = self.sampling(mu, log_var)
        reco = self.decoder(z)
        loss_gen, reco_loss, KL_loss = loss_function(reco, data, mu, log_var,
                                                     reduction=reduction, beta=self.beta, decoder_type=self.decoder_type)
        return reco, z, mu, log_var, loss_gen, reco_loss, KL_loss, 0

class iVAE(nn.Module):
    def __init__(self, lr_svi, x_dim, z_dim, h_dim1=512, h_dim2=256, cuda=True,
                 activation='tanh', svi_optimizer='ADAM', beta=1, decoder_type='bernoulli'):
        super(iVAE, self).__init__()

        self.decoder_type = decoder_type
        self.beta = beta
        self.to_cuda = cuda
        self.lr_svi=lr_svi
        if svi_optimizer == 'ADAM':
            self.optimizer = torch.optim.Adam

        # encoder part
        self.ff1 = nn.Linear(x_dim, h_dim1)
        self.ff2 = nn.Linear(h_dim1, h_dim2)
        self.ff3_mu = nn.Linear(h_dim2, z_dim)
        self.ff3_var = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fb4 = nn.Linear(z_dim, h_dim2)
        self.fb5 = nn.Linear(h_dim2, h_dim1)
        self.fb6 = nn.Linear(h_dim1, x_dim)

        if activation == 'tanh':
            self.activ_func = torch.nn.Tanh()
        elif activation == 'relu':
            self.activ_func = torch.nn.ReLU()

        self.enc = nn.Sequential(self.ff1, self.activ_func, self.ff2, self.activ_func)
        self.dec = nn.Sequential(self.fb4, self.activ_func, self.fb5, self.activ_func)

        # self.fb6_mu = nn.Linear(h_dim1, x_dim)
        # self.fb6_var = nn.Linear(h_dim1, x_dim)

        self.param_enc = [self.ff1.weight, self.ff1.bias, self.ff2.weight, self.ff2.bias,
                          self.ff3_mu.weight, self.ff3_mu.bias, self.ff3_var.weight, self.ff3_var.bias]

        self.param_dec = [self.fb4.weight, self.fb4.bias, self.fb5.weight, self.fb5.bias,
                          self.fb6.weight, self.fb6.bias]

        self.z_dim = z_dim

    def encoder(self, x):
        #h = torch.tanh(self.ff1(x))
        #h = torch.tanh(self.ff2(h))
        h = self.enc(x)
        mu = self.ff3_mu(h)
        log_var = self.ff3_var(h)
        return mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def decoder(self, z):
        #h = torch.tanh(self.fb4(z))
        #h = torch.tanh(self.fb5(h))
        h = self.dec(z)

        if self.decoder_type == 'bernoulli':
            to_return = torch.sigmoid(self.fb6(h))
        elif self.decoder_type == 'gaussian':
            to_return = self.fb6(h)
        return to_return

    def forward(self, x, nb_it):
        phi = PHI()
        if self.to_cuda:
            phi = phi.cuda()
        param_svi = list(phi.parameters())
        optimizer_SVI = self.optimizer(phi.parameters(), lr=self.lr_svi)
        #optimizer_SVI = torch.optim.Adam

        phi.mu_p.data, phi.log_var_p.data = self.encoder(torch.flatten(x, start_dim=1))

        ## Iterative refinement of the posterior
        enable_grad(param_svi)
        for idx_it in range(nb_it):
            optimizer_SVI.zero_grad()
            _, _, loss_gen, _, _ = self.step(x, phi)
            optimizer_SVI.step()
        disable_grad(param_svi)

        return phi

    def forward_eval(self, x, nb_it, freq_extra=0, reduction='sum', x_clear=None):
        phi = PHI()
        if self.to_cuda:
            phi = phi.cuda()
        param_svi = list(phi.parameters())
        optimizer_SVI = torch.optim.Adam(phi.parameters(), lr=self.lr_svi)

        phi.mu_p.data, phi.log_var_p.data = self.encoder(torch.flatten(x, start_dim=1))
        if freq_extra != 0:
            reco_l, _, _, z_l, loss_gen_l, reco_loss_l, KL_loss_l, nb_it_l = [],[],[],[],[],[],[],[]
            mu_l = torch.zeros(x.size(0),(nb_it//freq_extra)+1,self.z_dim).cuda()
            log_var_l = torch.zeros(x.size(0),(nb_it//freq_extra)+1,self.z_dim).cuda()
        ## Iterative refinement of the posterior
        enable_grad(param_svi)
        torch.set_printoptions(precision=10)
        idx_freq = 0
        for idx_it in range(nb_it):

            optimizer_SVI.zero_grad()
            reco, z, loss_gen, reco_loss, KL_loss = self.step(x, phi, reduction=reduction, x_clear=x_clear)

            optimizer_SVI.step()

            if (freq_extra!=0) and ((idx_it%freq_extra== 0) or idx_it==nb_it-1):
                #print(phi.mu_p.data[4,:])
                reco_l.append(reco.data)
                z_l.append(z.data)
                mu_l[:,idx_freq,:] = phi.mu_p.data
                log_var_l[:, idx_freq, :] = phi.log_var_p.data
                loss_gen_l.append(loss_gen.data)
                reco_loss_l.append(reco_loss.data)
                KL_loss_l.append(KL_loss.data)
                nb_it_l.append(idx_it)
                idx_freq+=1

        disable_grad(param_svi)

        if freq_extra != 0:
            reco_l = torch.stack(reco_l, 1)
            z_l = torch.stack(z_l, 1)
            loss_gen_l = torch.stack(loss_gen_l, 0)
            reco_loss_l = torch.stack(reco_loss_l, 0)
            KL_loss_l = torch.stack(KL_loss_l, 0)
            nb_it_l = torch.tensor(nb_it_l)
            return reco_l, z_l, mu_l, log_var_l, loss_gen_l, reco_loss_l, KL_loss_l, nb_it_l
        
        else:
            return reco, z, phi.mu_p.data, phi.log_var_p.data, loss_gen, reco_loss, KL_loss, 0

    def step(self, x, phi=None, mu=None, log_var=None, reduction='sum', x_clear=None):
        
        if phi is not None:
            z = self.sampling(phi.mu_p, phi.log_var_p)
            reco = self.decoder(z)
            loss_gen, reco_loss, KL_loss = loss_function(reco, x, phi.mu_p, phi.log_var_p,
                                                         reduction=reduction, beta=self.beta,
                                                         decoder_type=self.decoder_type, x_clear=x_clear)
        else:
            z = self.sampling(mu, log_var)
            reco = self.decoder(z)
            loss_gen, reco_loss, KL_loss = loss_function(reco, x, mu, log_var,
                                                         reduction=reduction, beta=self.beta,
                                                         decoder_type= self.decoder_type, x_clear=x_clear)
        loss_gen.backward()

        return reco, z, loss_gen, reco_loss, KL_loss
    


class PHI(nn.Module):
    def __init__(self):
        super(PHI, self).__init__()
        self.log_var_p = nn.Parameter() 
        self.mu_p = nn.Parameter()

        self.log_var = 0 
        self.mu = 0
    

        
#def forward(self, x):
#    mu_p, log_var_p = self.encoder(x.view(-1, 784))
#    z = self.sampling(mu_p, log_var_p)
#    x_r = self.decoder(z)
#    # mu_l, log_var_l = self.decoder(z)

#    # x_r = self.sampling(mu_l, log_var_l)

#    return x_r.view_as(x), (mu_p, log_var_p), (mu_l, log_var_l)

def loss_function(recon_x, x, mu_p, log_var_p, reduction='mean', beta=1, decoder_type='bernoulli', x_clear=None):
    ''' VAE loss function '''

    if decoder_type == 'bernoulli':
        reco = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='none').sum(-1)
    elif decoder_type == 'gaussian':
        if x_clear is None:
            reco = F.mse_loss(recon_x, x.view(-1, 784), reduction='none').sum(-1)
        else:
            reco = F.mse_loss(recon_x, x_clear.view(-1, 784), reduction='none').sum(-1)

    #reco = F.mse_loss(recon_x, x.view_as(recon_x), reduction='sum')
    KLD =  - beta * 0.5 * torch.sum(1 + log_var_p - mu_p.pow(2) - log_var_p.exp(), -1)

    
    # print(reco.shape)
    # print(KLD.shape)

    if reduction == 'mean':
        reco = reco.mean()
        KLD = KLD.mean()
    elif reduction == 'sum':
        reco = reco.sum()
        KLD = KLD.sum()
    
    
    total_loss = reco + KLD
    return total_loss, reco, KLD


def enable_grad(param_group):
    for p in param_group:
        p.requires_grad = True


def disable_grad(param_group):
    for p in param_group:
        p.requires_grad = False