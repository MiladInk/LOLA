from typing import Dict, List

import torch
from torch import autograd


def normalize(t: torch.tensor) -> torch.tensor:
    norm = torch.sqrt((t ** 2).sum())
    return t / norm


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

        V1_taylor = torch.dot(dV1_wrt_player2_theta.detach(), dV2_wrt_player2_theta)

        dV1_taylor_wrt_player1_theta_list = autograd.grad(
            outputs=V1_taylor,
            inputs=self.theta_list,
            retain_graph=True)

        for t, dV1_wrt_t, dV1_taylor_wrt_t in zip(self.theta_list,
                                                  dV1_wrt_player1_theta_list,
                                                  dV1_taylor_wrt_player1_theta_list):
            t.data.add_(dV1_wrt_t * self.lr + dV1_taylor_wrt_t * self.eta)


class NormLOLAAgent(Agent):
    def __init__(self, parameters: Dict[str, torch.tensor], lr: float, eta: float):
        super(NormLOLAAgent, self).__init__(parameters=parameters)
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

        dV1_taylor_wrt_player1_theta_list = autograd.grad(
            outputs=V1_taylor,
            inputs=self.theta_list,
            retain_graph=True)

        for t, dV1_wrt_t, dV1_taylor_wrt_t in zip(self.theta_list,
                                                  dV1_wrt_player1_theta_list,
                                                  dV1_taylor_wrt_player1_theta_list):
            t.data.add_(dV1_wrt_t * self.lr + dV1_taylor_wrt_t * self.eta)


def point_normalize(t: torch.tensor) -> torch.tensor:
    return t.true_divide(t + 1e-8)


class PointNormLOLAAgent(Agent):
    def __init__(self, parameters: Dict[str, torch.tensor], lr: float, eta: float):
        super(PointNormLOLAAgent, self).__init__(parameters=parameters)
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

        V1_taylor = torch.dot(point_normalize(dV1_wrt_player2_theta), point_normalize(dV2_wrt_player2_theta))

        dV1_taylor_wrt_player1_theta_list = autograd.grad(
            outputs=V1_taylor,
            inputs=self.theta_list,
            retain_graph=True)

        for t, dV1_wrt_t, dV1_taylor_wrt_t in zip(self.theta_list,
                                                  dV1_wrt_player1_theta_list,
                                                  dV1_taylor_wrt_player1_theta_list):
            t.data.add_(dV1_wrt_t * self.lr + dV1_taylor_wrt_t * self.eta)


class MySOSAgent(Agent):
    def __init__(self, parameters: Dict[str, torch.tensor], lr: float, alpha: float, a: float, b: float):
        super(MySOSAgent, self).__init__(parameters=parameters)
        self.lr = lr
        self.alpha = alpha
        self.a = a
        self.b = b

    def update_parameters(self, V1, V2, player2):
        L1, L2 = -V1, -V2

        dL1_wrt_player1_theta_list = autograd.grad(
            outputs=L1,
            inputs=self.theta_list,
            create_graph=True,
            retain_graph=True)
        dL1_wrt_player1_theta = torch.cat(dL1_wrt_player1_theta_list)

        dL2_wrt_player2_theta_list = autograd.grad(
            outputs=L2,
            inputs=player2.theta_list,
            create_graph=True,
            retain_graph=True)
        dL2_wrt_player2_theta = torch.cat(dL2_wrt_player2_theta_list)

        dL1_wrt_player2_theta_list = autograd.grad(
            outputs=L1,
            inputs=player2.theta_list,
            create_graph=True,
            retain_graph=True)
        dL1_wrt_player2_theta = torch.cat(dL1_wrt_player2_theta_list)

        dL2_wrt_player1_theta_list = autograd.grad(
            outputs=L2,
            inputs=self.theta_list,
            create_graph=True,
            retain_graph=True)
        dL2_wrt_player1_theta = torch.cat(dL2_wrt_player1_theta_list)

        term_player1 = torch.dot(dL1_wrt_player2_theta, dL2_wrt_player2_theta.detach())
        d_term_player1_wrt_player1_theta_list = autograd.grad(
            outputs=term_player1,
            inputs=self.theta_list,
            create_graph=True,
            retain_graph=True,
        )
        d_term_player1_wrt_player1_theta = torch.cat(d_term_player1_wrt_player1_theta_list)

        term_player2 = torch.dot(dL2_wrt_player1_theta, dL1_wrt_player1_theta.detach())
        d_term_player2_wrt_player2_theta_list = autograd.grad(
            outputs=term_player2,
            inputs=player2.theta_list,
            create_graph=True,
            retain_graph=True,
        )
        d_term_player2_wrt_player2_theta = torch.cat(d_term_player2_wrt_player2_theta_list)

        xi_0_player1 = dL1_wrt_player1_theta - self.alpha * d_term_player1_wrt_player1_theta
        xi_0_player2 = dL2_wrt_player2_theta - self.alpha * d_term_player2_wrt_player2_theta
        xi_0 = [xi_0_player1, xi_0_player2]

        chi_term_player1 = torch.dot(dL1_wrt_player2_theta.detach(), dL2_wrt_player2_theta)
        d_chi_term_player1_wrt_player1_theta_list = autograd.grad(
            outputs=chi_term_player1,
            inputs=self.theta_list,
            create_graph=True,
            retain_graph=True,
        )
        d_chi_term_player1_wrt_player1_theta = torch.cat(d_chi_term_player1_wrt_player1_theta_list)

        chi_term_player2 = torch.dot(dL2_wrt_player1_theta.detach(), dL1_wrt_player1_theta)
        d_chi_term_player2_wrt_player2_theta_list = autograd.grad(
            outputs=chi_term_player2,
            inputs=player2.theta_list,
            create_graph=True,
            retain_graph=True,
        )
        d_chi_term_player2_wrt_player2_theta = torch.cat(d_chi_term_player2_wrt_player2_theta_list)
        chi = [d_chi_term_player1_wrt_player1_theta, d_chi_term_player2_wrt_player2_theta]

        dot = torch.dot(-self.alpha * torch.cat(chi), torch.cat(xi_0))
        # noinspection PyTypeChecker
        p1 = 1 if dot >= 0 else min(1, -self.a * torch.norm(torch.cat(xi_0)) ** 2 / dot)
        xi = torch.cat([dL1_wrt_player1_theta, dL2_wrt_player2_theta])
        xi_norm = torch.norm(xi)
        p2 = xi_norm ** 2 if xi_norm < self.b else 1
        p = min(p1, p2)

        for t, dL1_wrt_t, d_term_wrt_t, d_chi_term_wrt_t in zip(self.theta_list,
                                                                dL1_wrt_player1_theta_list,
                                                                d_term_player1_wrt_player1_theta_list,
                                                                d_chi_term_player1_wrt_player1_theta_list):
            t.data.add_(-(dL1_wrt_t * self.lr - d_term_wrt_t * self.alpha - p * self.alpha * d_chi_term_wrt_t))


class ColabSOSAgent(Agent):
  def __init__(self, parameters: Dict[str, torch.tensor], lr: float, alpha: float, a: float = 0.5, b: float = 0.1):
    super(ColabSOSAgent, self).__init__(parameters=parameters)
    self.lr = lr
    self.alpha = alpha
    self.a = a
    self.b = b

  def update_parameters(self, V1, V2, player2):
    L1 = -V1
    L2 = -V2
    n = 2
    losses = [L1, L2]
    th = [self.theta_list, player2.theta_list]

    def get_gradient(function, param):
      grad_list = torch.autograd.grad(function, param, create_graph=True)
      grad = torch.cat(grad_list)
      return grad

    grad_L = [[get_gradient(losses[j], th[i]) for j in range(n)] for i in range(n)]

    terms = [sum([torch.dot(grad_L[j][i], grad_L[j][j].detach())
                  for j in range(n) if j != i]) for i in range(n)]
    xi_0 = [grad_L[i][i] - self.alpha * get_gradient(terms[i], th[i]) for i in range(n)]
    chi = [get_gradient(sum([torch.dot(grad_L[j][i].detach(), grad_L[j][j])
                             for j in range(n) if j != i]), th[i]) for i in range(n)]
    # Compute p
    dot = torch.dot(-self.alpha * torch.cat(chi), torch.cat(xi_0))
    # noinspection PyTypeChecker
    p1 = 1 if dot >= 0 else min(1, -self.a * torch.norm(torch.cat(xi_0)) ** 2 / dot)
    xi = torch.cat([grad_L[i][i] for i in range(n)])
    xi_norm = torch.norm(xi)
    p2 = xi_norm ** 2 if xi_norm < self.b else 1
    p = min(p1, p2)
    print('p is', p)
    grads = [xi_0[i] - p * self.alpha * chi[i] for i in range(n)]

    th_updated = torch.cat(th[0]) - self.lr * grads[0]

    def get_len(v):
        return v.view(-1).shape[0]

    start = 0
    for t in self.theta_list:
        t.data = th_updated[start: start+get_len(t)]
        start += get_len(t)
