import torch
import torch.nn as nn
import torch.nn.functional as F



class NoiseMapper(nn.Module):
    def __init__(self, inputdim=512):
        super(NoiseMapper, self).__init__()
        self.linear1 = torch.nn.Linear(inputdim, 1024)
        self.linear2 = torch.nn.Linear(1024, 1024)
        self.linear3 = torch.nn.Linear(1024, 1024)
        self.linear4 = torch.nn.Linear(1024, inputdim)
        self.linear5 = torch.nn.Linear(inputdim, 1024)
        self.linear6 = torch.nn.Linear(1024, 1024)
        self.linear7 = torch.nn.Linear(1024, 1024)
        self.linear8 = torch.nn.Linear(1024, inputdim)

    def forward(self, x):
        mu = F.leaky_relu(self.linear1(x))
        mu = F.leaky_relu(self.linear2(mu))
        mu = F.leaky_relu(self.linear3(mu))
        mu = self.linear4(mu)
        std = F.leaky_relu(self.linear5(x))
        std = F.leaky_relu(self.linear6(std))
        std = F.leaky_relu(self.linear7(std))
        std = self.linear8(std)
        return mu + std.exp()*(torch.randn(mu.shape).to(x.device))
    
    def loss(self, real, fake, temp=0.1, lam=0.5):
        sim = torch.cosine_similarity(real.unsqueeze(1), fake.unsqueeze(0), dim=-1)
        if temp > 0.:
            sim = torch.exp(sim/temp)
            sim1 = torch.diagonal(F.softmax(sim, dim=1))*temp
            sim2 = torch.diagonal(F.softmax(sim, dim=0))*temp
            if 0.<lam < 1.:
                return -(lam*torch.log(sim1) + (1.-lam)*torch.log(sim2))
            elif lam == 0:
                return -torch.log(sim2)
            else:
                return -torch.log(sim1)
        else:
            return -torch.diagonal(sim)









class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
        m_layer_features = 512, 
        m_num_layers = 0,                 
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        self.m_layer_features = m_layer_features

        self.mani = None
        self.synthesis_kwargs = synthesis_kwargs
        self.mapping_kwargs = mapping_kwargs

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, fts=None, styles=None, return_styles=False, step=1, w=None, return_w=False, **synthesis_kwargs):
        if w is not None:
            ws = w
        else:
            ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        if fts is None:
            fts = torch.randn(z.size()[0], self.m_layer_features).to(z.device)
            fts = fts/fts.norm(dim=-1, keepdim=True)
        if return_styles:
            img, styles = self.synthesis(ws, fts=fts, styles=styles, return_styles=return_styles, **synthesis_kwargs)
        else:
            img = self.synthesis(ws, fts=fts, styles=styles, return_styles=return_styles, **synthesis_kwargs)
            
        if return_w and return_styles:
            return img, ws, styles
        elif return_w and not return_styles:
            return img, ws
        elif not return_w and return_styles:
            return img, styles
        else:
            return img