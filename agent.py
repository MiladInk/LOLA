from typing import Dict, List

import torch
from torch import autograd

from lola import normalize


class AgentUpdateStrategy:
  pass


class Agent:
  def __init__(self, parameters: Dict[str, torch.tensor]):
    self.parameters = parameters

  @property
  def theta_list(self) -> List[torch.tensor]:
    return [param_value for param_value in self.parameters.values()]

  def update_parameters(self, agent_reward, player2_reward, player2):
    raise NotImplementedError


class LOLAAgent(Agent):
  def __init__(self, parameters: Dict[str, torch.tensor], lr: float, eta: float):
    super(LOLAAgent, self).__init__(parameters=parameters)
    self.lr = lr
    self.eta = eta

  def update_parameters(self, V1, V2, player2):
    dV1_wrt_player1_theta_list = autograd.grad(
      outputs=V1,
      inputs=self.theta_list,
      create_graph=True,
      retain_graph=True)

    dV2_wrt_player2_theta_list = autograd.grad(
      outputs=V2,
      inputs=player2.theta_list,
      create_graph=True,
      retain_graph=True)
    dV2_wrt_player2_theta = torch.cat(dV2_wrt_player2_theta_list)

    dV1_wrt_player2_theta_list = autograd.grad(
      outputs=V1,
      inputs=player2.theta_list,
      create_graph=True,
      retain_graph=True)
    dV1_wrt_player2_theta = torch.cat(dV1_wrt_player2_theta_list)

    V1_taylor = torch.dot(normalize(dV1_wrt_player2_theta), normalize(dV2_wrt_player2_theta))

    # print('V_taylor %.4f' % V1_taylor)

    # V1_taylor = torch.dot(dV1_wrt_player2_theta.detach(), dV2_wrt_player2_theta)

    dV1_taylor_wrt_player1_theta_list = autograd.grad(
      outputs=V1_taylor,
      inputs=self.theta_list,
      retain_graph=True)

    for t, dV1_wrt_t, dV1_taylor_wrt_t in zip(self.theta_list,
                                              dV1_wrt_player1_theta_list,
                                              dV1_taylor_wrt_player1_theta_list):

      t.data.add_(dV1_wrt_t * self.lr + dV1_taylor_wrt_t * self.eta)

