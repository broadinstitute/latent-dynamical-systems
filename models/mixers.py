import torch
from torch import nn
import numpy as np

class NMFMixer(nn.Module):
    def __init__(self, latent_dim, out_dim,reg_weight = 1e-5,**kwargs) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.ldecode = nn.Parameter(torch.randn(latent_dim, out_dim), requires_grad=True)
        self.reg_weight = reg_weight

    def build_params(self):
        self.decode = 0.1* torch.square(self.ldecode)

    def decoder(self,x):
        return x@self.decode

    def reg(self):
        d = self.decode/torch.norm(self.decode, dim= 1, keepdim = True)
        return self.reg_weight*torch.abs((1-torch.eye(self.latent_dim,device = 'cuda'))* (d@d.T)).mean()


class MLPMixer(nn.Module):
    def __init__(self, latent_dim, out_dim,reg_weight = 1e-5,hidden_dim = 200, nonlinearity = 'softplus',bias = True, dropout_prob = 0.0, **kwargs) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        if nonlinearity=='softplus':
            nl = nn.Softplus()
        elif nonlinearity=='relu':
            nl = nn.ReLU()
        elif nonlinearity=='celu':
            nl = nn.CELU()

        dropout = nn.Dropout(dropout_prob) 

        if bias==False:
            self.decoder = nn.Sequential(nn.Linear(latent_dim,hidden_dim,bias = None), nl, dropout, nn.Linear(hidden_dim,hidden_dim,bias = None),nl, dropout, nn.Linear(hidden_dim,out_dim,bias = None),nn.Softplus())
        else:
            self.decoder = nn.Sequential(nn.Linear(latent_dim,hidden_dim), nl, dropout, nn.Linear(hidden_dim,hidden_dim), nl, dropout, nn.Linear(hidden_dim,out_dim),nl) #note the terminal nonlinearity

    def build_params(self):
        pass

    def reg(self):
        return 0.0
    
class JointSequenceLatentMixer(nn.Module):
    def __init__(self, latent_dim, reg_weight = 1e-5, **kwargs) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        nl = nn.ReLU()
        self.decoder_nn = nn.Sequential(nn.Linear(2*latent_dim,200,bias = None), nl, nn.Linear(200,200,bias = None),nl,nn.Linear(200,1,bias = None),nn.Softplus())
        self.bpnet = BPNet(n_filters = 8, n_layers = 4, n_outputs = 2*latent_dim,n_control_tracks = 0)
        
    def build_params(self):
        pass
    def reg(self):
        return 0.0
    def decoder(self,latents, gene_seqs,return_attn = False):
        embeds = self.bpnet(gene_seqs) # 
        embed_k = embeds[:,:self.latent_dim]
        embed_v = embeds[:,self.latent_dim:]
        attention = torch.softmax((latents[:,:,None]*embed_k).sum(axis = 1)/20,dim = -1)
        z = torch.cat([(attention[:,None,:]*embed_v).sum(axis = -1), latents],axis = 1)
        if return_attn:
            return self.decoder_nn(z).squeeze(),attention,embed_k,latents
        else:
            return self.decoder_nn(z).squeeze()

        
    
class ReLUMLPMixer(MLPMixer):
    def __init__(self, latent_dim, out_dim, reg_weight=0.00001, bias=True, **kwargs) -> None:
        super().__init__(latent_dim, out_dim, reg_weight,  bias = bias,nonlinearity='relu', **kwargs)
    
    def forward(self,x):
        return (1+torch.tanh(4*x))*super().forward(x) #this is a bug but forward is not used


class SeqKernelMLPMixer(ReLUMLPMixer):
    def __init__(self, latent_dim, out_dim, path, reg_weight=0.00001, syntax_weight = 1e-4, num_neighbors= 20, **kwargs) -> None:
        super().__init__(latent_dim, out_dim, reg_weight, **kwargs)
        self.register_buffer('knn',torch.from_numpy(np.load(path)[:,:(num_neighbors)]).long())
        self.num_gene = self.knn.shape[0]
        self.nn= self.knn.shape[1]
        self.syntax_weight = syntax_weight
        
    def reg(self):
        reg = super().reg()
        gene_loadings = self.decoder[-2].weight #num_gene x latent_dim, -1 is the softplus
        normalized_gene_loadings = gene_loadings/(1e-6+torch.norm(gene_loadings, dim = 1,keepdim = True))
        gene_feats = normalized_gene_loadings[self.knn.flatten()].view(self.num_gene, self.nn, -1) #num gene x num neighbors x latent_di[m
        syn_reg = torch.square(normalized_gene_loadings[:,None,:]-gene_feats).sum(dim = -1).mean(dim = -1).mean()
        return reg + self.syntax_weight*syn_reg





class SeqKernelNMFMixer(NMFMixer):
    def __init__(self, latent_dim, out_dim, reg_weight=0.00001, syntax_weight = 1e-4, num_neighbors= 20, **kwargs) -> None:
        super().__init__(latent_dim, out_dim, reg_weight, **kwargs)
        self.register_buffer('knn',torch.from_numpy(np.load('gene_seq_knn.npy')[:,:(num_neighbors)]).long())
        self.num_gene = self.knn.shape[0]
        self.nn= self.knn.shape[1]
        self.syntax_weight = syntax_weight
        
    def reg(self):
        nmf_reg = super().reg()
        gene_loadings = self.decode.T #num_gene x latent_dim
        normalized_gene_loadings = gene_loadings/(1e-6+torch.norm(gene_loadings, dim = 1,keepdim = True))
        gene_feats = normalized_gene_loadings[self.knn.flatten()].view(self.num_gene, self.nn, -1) #num gene x num neighbors x latent_di[m
        syn_reg = torch.square(normalized_gene_loadings[:,None,:]-gene_feats).sum(dim = -1).mean(dim = -1).mean()
        return nmf_reg + self.syntax_weight*syn_reg


class SeqPredNMFMixer(NMFMixer):
    def __init__(self, latent_dim, out_dim, reg_weight=0.00001, syntax_weight = 1e-4, num_neighbors= 20, **kwargs) -> None:
        super().__init__(latent_dim, out_dim, reg_weight, **kwargs)
        self.register_buffer('gene_features',torch.from_numpy(np.load('gene_features.npy')).float())
        self.num_gene = self.gene_features.shape[0]
        self.syntax_weight = syntax_weight
        self.predictor = nn.Linear(latent_dim,self.gene_features.shape[1])
        
    def reg(self):
        nmf_reg = super().reg()
        gene_loadings = self.decode.T #num_gene x latent_dim
        syn_reg = torch.square(self.gene_features- self.predictor(gene_loadings)).mean()
        return nmf_reg + self.syntax_weight*syn_reg



class SeqPredMLPMixer(ReLUMLPMixer):
    def __init__(self, latent_dim, out_dim, gene_features, reg_weight=0.00001, syntax_weight = 1e-4, num_neighbors= 20, **kwargs) -> None:
        super().__init__(latent_dim, out_dim, reg_weight, **kwargs)
        self.register_buffer('gene_features',torch.from_numpy(gene_features).float())
        self.num_gene = self.gene_features.shape[0]
        self.syntax_weight = syntax_weight
        self.predictor = nn.Sequential(nn.Linear(200,200),nn.ReLU(), nn.Linear(200,200), nn.ReLU(),nn.Linear(200,self.gene_features.shape[1]))
        
    def reg(self):
        reg = super().reg()
        gene_loadings = self.decoder[-2].weight #num_gene x latent_dim, -1 is the softplus
        syn_reg = torch.square(self.gene_features- self.predictor(gene_loadings)).mean()
        return reg + self.syntax_weight*syn_reg
    




class SeqAttentionMLPMixer(nn.Module):
    def __init__(self, latent_dim, out_dim, gene_features, reg_weight=0.00001, syntax_weight = 1e-4, num_neighbors= 20, **kwargs) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer('gene_features',torch.from_numpy(gene_features).float())
        self.num_gene = self.gene_features.shape[0]
        self.use = nn.Parameter(torch.tensor(0,dtype = float),requires_grad = True)
        self.kv = nn.Linear(self.gene_features.shape[1], 1 + 5)
        # self.kv = nn.Linear(self.gene_features.shape[1], latent_dim + 5)
        # self.q = nn.Sequential(nn.Linear(latent_dim, 128),nn.GELU(), nn.Linear(128, 5))
        self.q = nn.Sequential(nn.Linear(latent_dim, 5))
        self.decoder2 = nn.Sequential(nn.Linear(latent_dim,200), nn.ReLU(), nn.Linear(200,200),nn.ReLU(),nn.Linear(200,out_dim),nn.ReLU())

    def build_params(self):
        kv = self.kv(self.gene_features)
        self.k = kv[:,:5]
        self.v = 0.01*torch.square(kv[:,-1][None,:])
        # self.v = kv[:,-self.latent_dim:]
        
    
    def decoder(self,x):
        q = self.q(x)
        activity = torch.tanh(q@self.k.T/2.2) #batch x out_dim
        self.add_on = torch.sigmoid(self.use) * activity *self.v #self.v = lets the gene decide if it's going to be affected by the feature info
        return self.decoder2(x) + self.add_on

        # return self.decoder2(torch.softmax(q@self.k.T/2.2, dim = -1)@self.v)



    def reg(self):
        return 0
    


class GeneAttentionMLPMixer(nn.Module):
    def __init__(self, latent_dim, out_dim, gene_feature_file, reg_weight=0.00001, syntax_weight = 1e-4, num_neighbors= 20, **kwargs) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer('gene_features',torch.from_numpy(np.load(gene_feature_file)).float())
        self.num_gene = self.gene_features.shape[0]
        self.use = nn.Parameter(torch.tensor(-1,dtype = float),requires_grad = True)
        self.kv = nn.Linear(self.gene_features.shape[1], 1 + 5)
        self.q = nn.Sequential(nn.Linear(latent_dim, 5))
        self.decoder2 = nn.Sequential(nn.Linear(latent_dim,200), nn.Softplus(), nn.Linear(200,200),nn.Softplus(),nn.Linear(200,out_dim),nn.Softplus())

    def build_params(self):
        kv = self.kv(self.gene_features)
        self.k = kv[:,:5]
        self.v = 0.01*torch.square(kv[:,-1][None,:])
        
        
    
    def decoder(self,x):
        q = self.q(x)
        activity = torch.tanh(q@self.k.T/2.2) #batch x out_dim
        self.add_on = torch.sigmoid(self.use) * activity*self.v #self.v = gene specific scaling
        return self.decoder2(x) + self.add_on



    def reg(self):
        return 0    
    
class OldSeqAttentionMLPMixer(nn.Module):
    def __init__(self, latent_dim, out_dim, reg_weight=0.00001, syntax_weight = 1e-4, num_neighbors= 20, **kwargs) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer('gene_features',torch.from_numpy(np.load('gene_features.npy')).float())
        self.num_gene = self.gene_features.shape[0]
        self.syntax_weight = syntax_weight
        self.kv = nn.Linear(self.gene_features.shape[1], latent_dim + 5)
        self.q = nn.Linear(latent_dim, 5)
        self.decoder2 = nn.Sequential(nn.Linear(latent_dim,200), nn.Softplus(), nn.Linear(200,200),nn.Softplus(),nn.Linear(200,out_dim),nn.Softplus())

    def build_params(self):
        kv = self.kv(self.gene_features)
        self.k = kv[:,:5]
        self.v = kv[:,-self.latent_dim:]
        
        
    
    def decoder(self,x):
        q = self.q(x)
        attn = torch.softmax(q@self.k.T/2.2, dim = -1)
        self.add_on = self.syntax_weight * attn@self.v

        return self.decoder2(self.add_on + x)

    def reg(self):
        return 0

class NMFDecomposedMixer(nn.Module): ## assumes that genes are ordered as [tfs, targets]
    def __init__(self, latent_dim, hidden_dim, target_dim,reg1_weight = 1e-3,reg2_weight = 1e-2,reg3_weight = 1e-5, **kwargs) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.ldecode1 = nn.Parameter(torch.randn(latent_dim, hidden_dim), requires_grad=True)
        self.ldecode2 = nn.Parameter(torch.randn(hidden_dim, target_dim), requires_grad=True)
        self.ldecode3 = nn.Parameter(torch.randn(latent_dim, hidden_dim), requires_grad=True)
        self.reg1_weight = reg1_weight
        self.reg2_weight = reg2_weight
        self.reg3_weight = reg3_weight

    def get_decode1(self):
        assert self.decode1 is not None
        return self.decode1
    
    def get_decode2(self):
        assert self.decode2 is not None
        return self.decode2
    
    def get_decode3(self):
        assert self.decode3 is not None
        return self.decode3

    def build_params(self):
        self.decode1 = 0.1* torch.square(self.ldecode1)
        self.decode2 = 0.1* torch.square(self.ldecode2)
        self.decode3 = 0.1* torch.square(self.ldecode3)

        ## initialize decode2 with prior
        tf_target_prior = torch.from_numpy(pd.read_csv('/home/skambha6/chenlab/stnca/stnca_data/zebrafish/zebrafish_cisTarget/tf_target_spatial_df.csv', index_col=0).to_numpy()).float().cuda()
        
        self.decode2[tf_target_prior != 0] = 1

    def decoder(self,x):
        out_target = (x@self.decode1)@self.decode2
        out_tf = x@self.decode3
        return torch.cat([out_tf, out_target],dim = 1)

    def reg(self):
        d1 = self.decode1/torch.norm(self.decode1, dim= 1, keepdim = True)
        d2 = self.decode2/torch.norm(self.decode2, dim= 1, keepdim = True)
        d3 = self.decode3/torch.norm(self.decode3, dim= 1, keepdim = True)
        reg = self.reg1_weight*torch.abs((1-torch.eye(self.latent_dim,device = 'cuda'))* (d1@d1.T)).mean() + \
                    self.reg2_weight*torch.abs((1-torch.eye(self.hidden_dim,device = 'cuda'))* (d2@d2.T)).mean() + \
                        self.reg3_weight*torch.abs((1-torch.eye(self.latent_dim,device = 'cuda'))* (d3@d3.T)).mean()
        return reg
    

class BPNet(torch.nn.Module):
    """A basic BPNet model with stranded profile and total count prediction.
    """
    def __init__(self, n_filters=64, n_layers=8, n_outputs=2, 
    n_control_tracks=0, alpha=1, profile_output_bias=True, 
    count_output_bias=True, name=None, trimming=None, verbose=True):
        super(BPNet, self).__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.n_control_tracks = n_control_tracks

        self.alpha = alpha
        self.name = name or "bpnet.{}.{}".format(n_filters, n_layers)
        self.trimming = trimming or 2 ** n_layers

        self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
        self.irelu = torch.nn.ReLU()

        self.rconvs = torch.nn.ModuleList([
            torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2**i, 
                dilation=2**i) for i in range(1, self.n_layers+1)
        ])
        self.rrelus = torch.nn.ModuleList([
            torch.nn.ReLU() for i in range(1, self.n_layers+1)
        ])

        self.fconv = torch.nn.Conv1d(n_filters+n_control_tracks, n_outputs, 
            kernel_size=75, padding=37, bias=profile_output_bias)

        n_count_control = 1 if n_control_tracks > 0 else 0
        self.linear = torch.nn.Linear(n_filters+n_count_control, 1, 
            bias=count_output_bias)

    


    def forward(self, X, X_ctl=None):
        

        X = self.irelu(self.iconv(X))
        for i in range(self.n_layers):
            X_conv = self.rrelus[i](self.rconvs[i](X))
            X = torch.add(X, X_conv)

        if X_ctl is None:
            X_w_ctl = X
        else:
            X_w_ctl = torch.cat([X, X_ctl], dim=1)

        y_profile = self.fconv(X_w_ctl)

       
        return y_profile