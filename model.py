from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.models.torch.misc import normc_initializer, same_padding, SlimConv2d, SlimFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

from adversarial_comms.models.gnn import adversarialGraphML as gml_adv
from adversarial_comms.models.gnn import graphML as gml
from adversarial_comms.models.gnn import graphTools
import numpy as np
import copy

torch, nn = try_import_torch()
from torchsummary import summary

# https://ray.readthedocs.io/en/latest/using-ray-with-pytorch.html

DEFAULT_OPTIONS = {
    "activation": "relu",
    "agent_split": 0,
    "cnn_compression": 512,
    "cnn_filters": [[32, [8, 8], 4], [64, [4, 4], 2], [128, [4, 4], 2]],
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

class Model(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(model_config['custom_model_config'])

        #self.cfg = model_config['custom_options']
        self.n_agents = len(obs_space.original_space['agents'])
        self.activation = {
            'relu': nn.ReLU,
            'leakyrelu': nn.LeakyReLU
        }[self.cfg['activation']]

        obs_shape = obs_space.original_space['agents'][0]['obs'].shape
        state_shape = obs_space.original_space['agents'][0]['state'].shape

        #import pdb; pdb.set_trace()

        self.encoder = nn.Sequential(
            nn.Linear(obs_shape[0], 128),
            self.activation(),
            nn.Linear(128, 128),
            self.activation(),
            nn.Linear(128, self.cfg['graph_features']),
            self.activation()
        )
        summary(self.encoder, device="cpu", input_size=obs_shape)

        self.GFL = nn.Sequential(
            gml_adv.GraphFilterBatchGSOA(
                self.cfg['graph_features'],
                self.cfg['graph_features'],
                self.cfg['graph_tabs'],
                self.cfg['agent_split'],
                self.cfg['graph_edge_features'],
                False,
                forward_mode=self.cfg['forward_mode']
            ),
            self.activation()
        )

        logits_inp_features = self.cfg['graph_features']

        post_logits = [
            nn.Linear(logits_inp_features, 128),
            self.activation(),
            nn.Linear(128, 128),
            self.activation()
        ]
        self.outputs_per_agent = int(num_outputs/self.n_agents)
        logit_linear = nn.Linear(128, self.outputs_per_agent)
        nn.init.xavier_uniform_(logit_linear.weight)
        nn.init.constant_(logit_linear.bias, 0)
        post_logits.append(logit_linear)
        self.output = nn.Sequential(*post_logits)
        summary(self.output, device="cpu", input_size=(logits_inp_features,))

        #############
        self.value_encoder = nn.Sequential(
            nn.Linear(obs_shape[0]+state_shape[0], 128),
            self.activation(),
            nn.Linear(128, 128),
            self.activation(),
            nn.Linear(128, 128),
            self.activation(),
            nn.Linear(128, 128),
            self.activation(),
            nn.Linear(128, self.cfg['graph_features']),
            self.activation(),
        )
        summary(self.value_encoder, device="cpu", input_size=(obs_shape[0]+state_shape[0],))

        self.value_GFL = nn.Sequential(
            gml_adv.GraphFilterBatchGSOA(
                self.cfg['graph_features'],
                self.cfg['graph_features'],
                self.cfg['graph_tabs'],
                self.cfg['agent_split'],
                self.cfg['graph_edge_features'],
                False,
                forward_mode=self.cfg['forward_mode']
            ),
            self.activation()
        )

        value_post_logits = [
            nn.Linear(logits_inp_features, 128),
            self.activation(),
            nn.Linear(128, 128),
            self.activation(),
            nn.Linear(128, 128),
            self.activation(),
        ]
        logit_linear = nn.Linear(128, 1)
        nn.init.xavier_uniform_(logit_linear.weight)
        nn.init.constant_(logit_linear.bias, 0)
        value_post_logits.append(logit_linear)
        self.value_output = nn.Sequential(*value_post_logits)
        summary(self.value_output, device="cpu", input_size=(logits_inp_features,))

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        batch_size = input_dict["obs"]['gso'].shape[0]

        gso = input_dict["obs"]['gso'].unsqueeze(1)
        device = gso.device

        self.GFL[0].addGSO(gso)
        self.value_GFL[0].addGSO(torch.ones_like(gso).to(device))

        extract_feature_map = torch.zeros(batch_size, self.cfg['graph_features'], self.n_agents).to(device)
        value_extract_feature_map = torch.zeros(batch_size, self.cfg['graph_features'], self.n_agents).to(device)
        for i in range(self.n_agents):
            agent_obs = input_dict["obs"]['agents'][i]['obs']
            agent_state = input_dict["obs"]['agents'][i]['state']
            extract_feature_map[:, :, i] = self.encoder(agent_obs)
            value_extract_feature_map[:, :, i] = self.value_encoder(torch.cat([agent_obs, agent_state], dim=1))

        shared_feature = self.GFL(extract_feature_map)
        value_shared_feature = self.value_GFL(value_extract_feature_map)

        outputs = torch.empty(batch_size, self.n_agents, self.outputs_per_agent).to(device)
        values = torch.empty(batch_size, self.n_agents).to(device)

        for i in range(self.n_agents):
            outputs[:, i] = self.output(shared_feature[..., i])
            if self.cfg['forward_values']:
                values[:, i] = self.value_output(value_shared_feature[..., i]).squeeze(1)

        if self.cfg['forward_values']:
            self._cur_value = values

        if not torch.all(torch.isfinite(outputs)):
            import pdb; pdb.set_trace()

        return outputs.view(batch_size, self.n_agents*self.outputs_per_agent), state

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

