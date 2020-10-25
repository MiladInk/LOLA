import time

import torch
import torch.autograd as autograd
import matplotlib.pyplot as plt

alpha = 0.99


def get_initial_game_state(p1: torch.tensor, p2: torch.tensor):
  return torch.cat([p1 * p2, p1 * (1 - p2), (1 - p1) * p2, (1 - p1) * (1 - p2)], dim=0)


def get_transition_matrix(p1: torch.tensor, p2: torch.tensor):
  tm = torch.stack([p1 * p2, p1 * (1 - p2), (1 - p1) * p2, (1 - p1) * (1 - p2)], dim=1)
  return tm


def get_asymptotic_reward_mathematically(p1_init: torch.tensor, p2_init: torch.tensor, p1: torch.tensor,
                                         p2: torch.tensor, payoff1: torch.tensor, payoff2: torch.tensor):
  game_state = get_initial_game_state(p1_init, p2_init)
  t_matrix = get_transition_matrix(p1, p2)

  M = t_matrix.t() * alpha
  A = torch.inverse(torch.eye(4) - M)
  StateSum = torch.matmul(A, game_state)
  normalizing_factor = 1 / (1 - alpha)
  reward_1 = calculate_reward(payoff1, StateSum) / normalizing_factor
  reward_2 = calculate_reward(payoff2, StateSum) / normalizing_factor
  return reward_1, reward_2


def get_asymptotic_reward_iteratively(p1_init: torch.tensor, p2_init: torch.tensor, p1: torch.tensor,
                                      p2: torch.tensor, payoff1: torch.tensor, payoff2: torch.tensor):
  game_state = get_initial_game_state(p1_init, p2_init)
  t_matrix = get_transition_matrix(p1, p2)

  reward_1 = torch.zeros(1)
  reward_2 = torch.zeros(1)

  coef = 1.
  for i in range(20000):
    reward_1 = reward_1 + coef * calculate_reward(payoff1, game_state)
    reward_2 = reward_2 + coef * calculate_reward(payoff2, game_state)
    game_state = t_matrix.t().matmul(game_state)
    coef = coef * alpha
    print(reward_1)

  return reward_1, reward_2


def calculate_reward(payoff: torch.tensor, game_state: torch.tensor):
  return torch.sum(payoff.reshape(-1) * game_state)


if __name__ == '__main__':
  # payoff matrix assuming event payoff_player_x[player 1 choice][player 2 choice]
  IPD_payoff_player1 = torch.tensor([[-1, -3],
                                     [0, -2]])

  IPD_payoff_player2 = torch.tensor([[-1, 0],
                                     [-3, -2]])

  Chicken_payoff_player1 = torch.tensor([[0, -1],
                                        [10, -200]])

  Chicken_payoff_player2 = torch.tensor([[0, 10],
                                        [-1, -200]])

  player1_start_theta = torch.rand(1, requires_grad=True) - 0.5
  player2_start_theta = torch.rand(1, requires_grad=True) - 0.5
  player1_game_theta = torch.rand(4, requires_grad=True) - 0.5
  player2_game_theta = torch.rand(4, requires_grad=True) - 0.5


  # --- calculate the reward ---

  def players_rewards():
    # ---- players config ----

    player1_start_cooperate_probability = torch.sigmoid(player1_start_theta)
    player2_start_cooperate_probability = torch.sigmoid(player2_start_theta)
    # assuming CC, CD, DC, DD

    player1_cooperate_probability = torch.sigmoid(player1_game_theta)
    player2_cooperate_probability = torch.sigmoid(player2_game_theta)
    reward_1, reward_2 = get_asymptotic_reward_mathematically(player1_start_cooperate_probability,
                                                              player2_start_cooperate_probability,
                                                              player1_cooperate_probability,
                                                              player2_cooperate_probability,
                                                              Chicken_payoff_player1,
                                                              Chicken_payoff_player2)
    return reward_1, reward_2


  # --- lola ---- #

  for i in range(10000):
    V1, V2 = players_rewards()
    print('iter:', i, V1.item(), V2.item())

    dV1_wrt_player1_start_theta, dV1_wrt_player1_game_theta = autograd.grad(
      outputs=V1,
      inputs=[player1_start_theta, player1_game_theta],
      create_graph=True,
      retain_graph=True)

    dV2_wrt_player2_start_theta, dV2_wrt_player2_game_theta = autograd.grad(
      outputs=V2,
      inputs=[player2_start_theta, player2_game_theta],
      create_graph=True,
      retain_graph=True)

    dV1_wrt_player2_start_theta, dV1_wrt_player2_game_theta = autograd.grad(
      outputs=V1,
      inputs=[player2_start_theta, player2_game_theta],
      create_graph=True,
      retain_graph=True)

    dV2_wrt_player1_start_theta, dV2_wrt_player1_game_theta = autograd.grad(
      outputs=V2,
      inputs=[player1_start_theta, player1_game_theta],
      create_graph=True,
      retain_graph=True)

    V1_taylor = torch.dot(torch.cat([dV2_wrt_player2_start_theta, dV2_wrt_player2_game_theta]),
                          torch.cat([dV1_wrt_player2_start_theta.detach(), dV1_wrt_player2_game_theta.detach()]))

    V2_taylor = torch.dot(torch.cat([dV1_wrt_player1_start_theta, dV1_wrt_player1_game_theta]),
                          torch.cat([dV2_wrt_player1_start_theta.detach(), dV2_wrt_player1_game_theta.detach()]))

    dV1_taylor_wrt_player1_start_theta, dV1_taylor_wrt_player1_game_theta = autograd.grad(
      outputs=V1_taylor,
      inputs=[player1_start_theta, player1_game_theta],
      retain_graph=True)

    dV2_taylor_wrt_player2_start_theta, dV2_taylor_wrt_player2_game_theta = autograd.grad(
      outputs=V2_taylor,
      inputs=[player2_start_theta, player2_game_theta],
      retain_graph=True)

    lr = 3e-2
    eta = 3e-1 * 0

    if i > 2500:
      eta = 3e-1

    player1_start_theta.data.add_(dV1_wrt_player1_start_theta * lr + dV1_taylor_wrt_player1_start_theta * eta)
    player1_game_theta.data.add_(dV1_wrt_player1_game_theta * lr + dV1_taylor_wrt_player1_game_theta * eta)
    player2_start_theta.data.add_(dV2_wrt_player2_start_theta * lr + dV2_taylor_wrt_player2_start_theta * eta)
    player2_game_theta.data.add_(dV2_wrt_player2_game_theta * lr + dV2_taylor_wrt_player2_game_theta * eta)

    # plotting the probabilities
    if i % 100 == 0:
      player1_start_cooperate_probability = torch.sigmoid(player1_start_theta)
      player2_start_cooperate_probability = torch.sigmoid(player2_start_theta)
      # assuming CC, CD, DC, DD

      player1_cooperate_probability = torch.sigmoid(player1_game_theta)
      player2_cooperate_probability = torch.sigmoid(player2_game_theta)

      plt.bar(['p1CC', 'p1CD', 'p1DC', 'p1DD'], player1_cooperate_probability.tolist())
      plt.bar(['p2CC', 'p2CD', 'p2DC', 'p2DD'], player2_cooperate_probability.tolist())
      plt.bar(['p1_C', 'p2_C'], [player1_start_cooperate_probability.tolist()[0],
                                 player2_start_cooperate_probability.tolist()[0]])
      plt.show()
      time.sleep(1)
