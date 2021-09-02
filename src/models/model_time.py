from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor


class ModGNNConv(MessagePassing):
    def __init__(self, nn, aggr="mean", **kwargs):
        super(ModGNNConv, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        torch_geometric.nn.inits.reset(self.nn)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(x_j - x_i)

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)


class GNNBranch(nn.Module):
    def __init__(self, in_features, msg_features, out_features, activation):
        nn.Module.__init__(self)

        self.nns = self.generate_nn_instance(
            in_features, msg_features, out_features, activation
        )
        self.gnn = ModGNNConv(self.nns["gnn"], aggr="add")

    def generate_nn_instance(self, in_features, msg_features, out_features, activation):
        return torch.nn.ModuleDict(
            {
                "encoder": torch.nn.Sequential(
                    torch.nn.Linear(in_features, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, msg_features),
                ),
                "gnn": torch.nn.Sequential(
                    torch.nn.Linear(1 + msg_features, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 64),
                ),
                "post": torch.nn.Sequential(
                    torch.nn.Linear(64, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, out_features),
                ),
            }
        )

    def forward(
        self,
        p: torch.Tensor,
        x: torch.Tensor,
        time: torch.Tensor,
        comm_radius: torch.Tensor,
    ):
        assert x.ndim == 3  # batch and features
        assert p.ndim == 3  # batch and positions
        batch_size = x.shape[0]
        n_agents = x.shape[1]

        encoding_out = self.nns["encoder"](x)
        msg = torch.cat([encoding_out, time], dim=2)

        b = torch.arange(0, batch_size, dtype=torch.int64, device=x.device)
        batch = torch.repeat_interleave(b, n_agents)
        edge_index = torch_geometric.nn.pool.radius_graph(
            p.reshape(-1, p.shape[-1]), batch=batch, r=comm_radius[0], loop=True
        )
        gnn_in = msg.reshape(-1, msg.shape[-1])
        gnn_out = self.gnn(gnn_in, edge_index).view(batch_size, n_agents, -1)

        return self.nns["post"](gnn_out)


class Model(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **cfg):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.cfg = cfg
        self.n_agents = obs_space.original_space["pos"].shape[0]
        self.outputs_per_agent = int(num_outputs / self.n_agents)

        activation = {
            "relu": nn.ReLU,
            "leakyrelu": nn.LeakyReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }[self.cfg["activation"]]

        self.comm_range = torch.Tensor([self.cfg["comm_range"]])

        self.gnn = GNNBranch(
            6, self.cfg["msg_features"], self.outputs_per_agent, activation
        )
        self.gnn_value = GNNBranch(6, self.cfg["msg_features"], 1, activation)

        self.lookback_frames = int(
            self.cfg["comm_delay_mean"] + self.cfg["comm_delay_std"] * 3
        )
        self.view_requirements["prev_n_obs"] = ViewRequirement(
            data_col="obs", shift="-{}:0".format(self.lookback_frames), space=obs_space
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        batch_size = input_dict["obs"]["pos"].shape[0]
        device = input_dict["obs"]["pos"].device

        obs_time = restore_original_dimensions(
            input_dict["prev_n_obs"], self.obs_space, "torch"
        )
        indexes = (
            torch.normal(
                mean=torch.ones(batch_size, 1, self.n_agents, 1)
                * self.cfg["comm_delay_mean"],
                std=self.cfg["comm_delay_std"],
            )
            .clamp(0, self.lookback_frames)
            .long()
            .to(device)
        )

        pos = torch.gather(obs_time["pos"], 1, indexes.repeat(1, 1, 1, 2)).squeeze(1)
        vel = torch.gather(obs_time["vel"], 1, indexes.repeat(1, 1, 1, 2)).squeeze(1)
        goal = torch.gather(obs_time["goal"], 1, indexes.repeat(1, 1, 1, 2)).squeeze(1)
        time = torch.gather(obs_time["time"], 1, indexes.repeat(1, 1, 1, 1)).squeeze(1)

        x = torch.cat([goal - pos, pos, pos + vel], dim=2)
        outputs = self.gnn(pos, x, time, self.comm_range)
        values = self.gnn_value(pos, x, time, self.comm_range)
        self._cur_value = values.view(-1, self.n_agents)

        return outputs.view(-1, self.n_agents * self.outputs_per_agent), state

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value
