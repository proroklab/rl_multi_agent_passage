from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.models.torch.misc import normc_initializer, same_padding, SlimConv2d, SlimFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

from GNN.Models.GNN import GNN
#from GNN.Models.Implementation.MRS import fpre_gen, fmid_gen, ffinal_gen, fmid_weights_gen

import numpy as np
import copy

torch, nn = try_import_torch()
from torchsummary import summary

# https://ray.readthedocs.io/en/latest/using-ray-with-pytorch.html

DEFAULT_OPTIONS = {
    "activation": "relu",
    "agent_split": 0,
    "cnn_compression": 32,
    "cnn_filters": [[8, [3, 3], 2], [16, [3, 3], 2], [32, [3, 3], 2], [32, [3, 3], 2]],
    "cnn_residual": False,
    "freeze_coop": True,
    "freeze_coop_value": False,
    "freeze_greedy": False,
    "freeze_greedy_value": False,
    "graph_edge_features": 1,
    "graph_features": 512,
    "graph_tabs": 3,
    "relative": True,
    "value_cnn_compression": 512,
    "value_cnn_filters": [[32, [8, 8], 2], [64, [4, 4], 2], [128, [4, 4], 2]],
    "forward_values": True,
    "forward_mode": "default"
}

def fpre_gen(indim=6, outdim=10, K=1):

    fpre_net = [nn.Sequential(
                nn.Linear(indim,64),
                nn.ReLU(),
                nn.Linear(64,64),
                nn.ReLU(),
                nn.Linear(64,outdim),
                nn.ReLU()
                ).double() for _ in range(K+1)]

    #summary(fpre_net[0].float(), device="cpu", input_size=(indim,))
    
    def fpre(X):
        batch, K, N, _, Dinput = X.shape; K-=1
        Yk = [None] * (K+1)
        for k in range(K+1):
            Xc = X[:,k,:,:,:].reshape(batch*N*N, Dinput)
            Yc = fpre_net[k](Xc)
            Dpre = Yc.shape[-1]
            Yk[k] = Yc.view(batch,N,N,Dpre)
        Y = torch.stack(Yk, dim=1)
        return Y

    return fpre, fpre_net



def fmid_gen(indim=10, outdim=10):

    fmid_net = nn.Sequential(
                nn.Linear(indim,64),
                nn.ReLU(),
                nn.Linear(64,64),
                nn.ReLU(),
                nn.Linear(64,outdim),
                nn.ReLU()
            ).double()

    def fmid(X):
        batch, K, N, Dpre = X.shape; K-=1
        Xc = X.view(batch*(K+1)*N, Dpre)
        Yc = fmid_net(Xc)
        Dmid = Yc.shape[-1]
        Y = Yc.view(batch,K+1,N,Dmid)
        return Y

    return fmid, fmid_net



def ffinal_gen(indim=10, outdim=3):

    ffinal_net = nn.Sequential(
                nn.Linear(indim,64),
                nn.ReLU(),
                nn.Linear(64,64),
                nn.ReLU(),
                nn.Linear(64,outdim),
            ).double()

    def ffinal(X):
        batch, N, Dmid = X.shape
        Xc = X.view(batch*N, Dmid)
        Yc = ffinal_net(Xc)
        Dfinal = Yc.shape[-1]
        Y = Yc.view(batch,N,Dfinal)
        return Y

    return ffinal, ffinal_net


class Model(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(model_config['custom_model_config'])

        #self.cfg = model_config['custom_options']
        self.n_agents = len(obs_space.original_space['agents'])
        self.outputs_per_agent = int(num_outputs/self.n_agents)

        self.activation = {
            'relu': nn.ReLU,
            'leakyrelu': nn.LeakyReLU
        }[self.cfg['activation']]

        self.obs_shape = obs_space.original_space['agents'][0]['obs'].shape
        #self.state_shape = obs_space.original_space['agents'][0]['state'].shape

        #import pdb; pdb.set_trace()


        K = self.cfg['graph_tabs']
        self.gnn = GNN(
            K=K,
            fpre=fpre_gen(self.obs_shape[0], 64, K=K),
            fmid=fmid_gen(64, 64),
            ffinal=ffinal_gen(64, self.outputs_per_agent)
        )

        #nn.init.xavier_uniform_(self.gnn.network[-1].ffinal.weight)
        #nn.init.constant_(self.gnn.network[-1].ffinal.bias, 0)

        #summary(self.output, device="cpu", input_size=(logits_inp_features,))

        #############

        self.gnn_value = GNN(
            K=K,
            fpre=fpre_gen(self.obs_shape[0], 64, K=K), #+state_shape[0], 10, K=K),
            fmid=fmid_gen(64, 64),
            ffinal=ffinal_gen(64, 1)
        )

        #nn.init.xavier_uniform_(self.gnn.network[-1].ffinal.weight)
        #nn.init.constant_(self.gnn.network[-1].ffinal.bias, 0)
        
        #summary(self.value_output, device="cpu", input_size=(logits_inp_features,))

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        batch_size = input_dict["obs"]['gso'].shape[0]
        #if batch_size > 1:
        #    print("FWD", batch_size)

        t_dim = self.cfg['graph_tabs']+1

        gso = input_dict["obs"]['gso'].unsqueeze(1).repeat(1, t_dim, 1, 1).permute(0,2,3,1)
        device = gso.device

        features = torch.zeros(batch_size, self.n_agents, self.obs_shape[0], t_dim).to(device)
        for i in range(self.n_agents):
            for j in range(t_dim):
                features[:, i, :, j] = input_dict["obs"]['agents'][i]['obs']


        outputs = self.gnn(gso, features)
        #import pdb; pdb.set_trace()        

        if self.cfg['forward_values']:
            values = self.gnn_value(gso, features)
            #import pdb; pdb.set_trace()
            self._cur_value = values.squeeze(2)

        if not torch.all(torch.isfinite(outputs)):
            import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        return outputs.view(batch_size, self.n_agents*self.outputs_per_agent), state

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

