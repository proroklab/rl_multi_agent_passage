from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

from ModGNN import GNN, GNNnode

# from GNN.Models.Implementation.MRS import fpre_gen, fmid_gen, ffinal_gen, fmid_weights_gen

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
    "forward_mode": "default",
}


class GNNnodeCustom(GNNnode):
    def __init__(self, K=1, indim=1, outdim=1, **kwargs):
        # Run GNNnode constructor
        super().__init__(K=K, **kwargs)
        # Define networks
        self.fpre_net = [
            nn.Sequential(
                nn.Linear(indim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            for _ in range(K + 1)
        ]

        self.fmid_net = [
            nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            for k in range(K + 1)
        ]
        self.ffinal_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, outdim),
        )
        # Register parameters
        self.register_params(self.fpre_net, "fpre")
        self.register_params(
            self.fmid_net, "fmid"
        )  # modules defined in lists/dicts must be registered (added to the GNNnode's list of params) in order to be learnable

    def fcom(self, A, X):
        # A: the adjacency matrix with shape [batch x N x N]
        # X: the joint state with shape [batch x N x Dinput]
        # output: the incoming data (dim=2) for each agent (dim=1), with shape [batch x N x N x Dinput]
        return (A[:, :, :, None] * X[:, None, :, :]) - (
            A[:, :, :, None] * X[:, :, None, :]
        )

    def fpre(self, X):
        # X: the joint state with shape [(batch*N*N) x (K+1) x Dinput]
        # output: the processed inputs of shape [(batch*N*N) x (K+1) x Dpre]
        return torch.stack(
            [self.fpre_net[k](X[:, k, :]) for k in range(X.shape[1])], dim=1
        )

    def fmid(self, X):
        # X: the joint state with shape [(batch*N) x (K+1) x Dpre]
        # output: the processed aggregated neighbourhoods of shape [(batch*N) x (K+1) x Dmid]
        return torch.stack(
            [self.fmid_net[k](X[:, k, :]) for k in range(X.shape[1])], dim=1
        )  # applies a different fmid to each neighbourhood

    def ffinal(self, X):
        # X: the joint state with shape [(batch*N) x Dmid]
        # output: the processed aggregated neighbourhoods of shape [(batch*N) x Dout]
        return self.ffinal_net(X)


class Model(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(model_config["custom_model_config"])

        # self.cfg = model_config['custom_options']
        self.n_agents = len(obs_space.original_space["agents"])
        self.outputs_per_agent = int(num_outputs / self.n_agents)

        self.activation = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU}[
            self.cfg["activation"]
        ]

        self.obs_shape = obs_space.original_space["agents"][0]["obs"].shape
        # self.state_shape = obs_space.original_space['agents'][0]['state'].shape

        # import pdb; pdb.set_trace()

        K = self.cfg["graph_tabs"]
        self.gnn = GNN(
            K=K,
            layers=[
                GNNnodeCustom(
                    K=K, indim=self.obs_shape[0], outdim=self.outputs_per_agent
                )
            ],
        )

        # nn.init.xavier_uniform_(self.gnn.network[-1].ffinal.weight)
        # nn.init.constant_(self.gnn.network[-1].ffinal.bias, 0)

        # summary(self.output, device="cpu", input_size=(logits_inp_features,))

        #############

        self.gnn_value = GNN(
            K=K, layers=[GNNnodeCustom(K=K, indim=self.obs_shape[0], outdim=1)]
        )

        # nn.init.xavier_uniform_(self.gnn.network[-1].ffinal.weight)
        # nn.init.constant_(self.gnn.network[-1].ffinal.bias, 0)

        # summary(self.value_output, device="cpu", input_size=(logits_inp_features,))

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        batch_size = input_dict["obs"]["gso"].shape[0]
        # if batch_size > 1:
        #    print("FWD", batch_size)

        t_dim = self.cfg["graph_tabs"] + 1

        gso = (
            input_dict["obs"]["gso"].unsqueeze(1).repeat(1, t_dim, 1, 1)
        )  # .permute(0,2,3,1)
        device = gso.device

        features = torch.zeros(batch_size, t_dim, self.n_agents, self.obs_shape[0]).to(
            device
        )
        for i in range(self.n_agents):
            for j in range(t_dim):
                features[:, j, i, :] = input_dict["obs"]["agents"][i]["obs"]

        outputs = self.gnn(gso, features)
        # import pdb; pdb.set_trace()

        if self.cfg["forward_values"]:
            values = self.gnn_value(gso, features)
            # import pdb; pdb.set_trace()
            self._cur_value = values.squeeze(2)

        if not torch.all(torch.isfinite(outputs)):
            import pdb

            pdb.set_trace()
        # import pdb; pdb.set_trace()
        return outputs.view(batch_size, self.n_agents * self.outputs_per_agent), state

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value
