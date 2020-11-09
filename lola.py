import random
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
  rescaling_factor = (1 - alpha)
  reward_1 = calculate_reward(payoff1, StateSum) * rescaling_factor
  reward_2 = calculate_reward(payoff2, StateSum) * rescaling_factor
  return reward_1, reward_2


def get_random_payoff_matrix():
  a = 0
  d = random.randint(a, 5)
  b = random.randint(-5, 5)
  c = random.randint(-5, 5)
  player1_payoff = [[a, b], [c, d]]
  player2_payoff = [[a, c], [b, d]]
  player1_payoff_tensor = torch.tensor(player1_payoff)
  player2_payoff_tensor = torch.tensor(player2_payoff)

  return player1_payoff_tensor, player2_payoff_tensor


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


def game_cooperate_probability(game_theta: torch.tensor) -> torch.tensor:
  return torch.sigmoid(game_theta)


def game_start_probability(start_theta: torch.tensor) -> torch.tensor:
  return torch.sigmoid(start_theta)


def normalize(t: torch.tensor) -> torch.tensor:
  norm = torch.sqrt((t**2).sum())
  return t/norm


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

  Exp1_payoff_player1 = torch.tensor([[0, 2.5],
                                      [-1., 1]])

  Exp1_payoff_player2 = torch.tensor([[0, -1],
                                      [2.5, 1]])

  rand_payoff_player1, rand_payoff_player2 = get_random_payoff_matrix()

  player1_start_theta = torch.rand(1, requires_grad=True) - 0.5
  player1_game_theta = torch.rand(4, requires_grad=True) - 0.5
  player2_start_theta = torch.rand(1, requires_grad=True) - 0.5
  player2_game_theta = torch.rand(4, requires_grad=True) - 0.5

  # player1_start_theta = torch.tensor([5.], requires_grad=True)
  # player1_game_theta = torch.tensor([5., -5., 5., -5.], requires_grad=True)
  # player2_start_theta = torch.tensor([5.], requires_grad=True)
  # player2_game_theta = torch.tensor([5., 5., -5., -5.], requires_grad=True)

  # player1_start_theta = torch.tensor([-5.], requires_grad=True)
  # player1_game_theta = torch.tensor([-5., -5., -5., -5.], requires_grad=True)
  # player2_start_theta = torch.tensor([-5.], requires_grad=True)
  # player2_game_theta = torch.tensor([-5., -5., -5., -5.], requires_grad=True)

  # --- calculate the reward ---

  chosen_payoff_player1 = Exp1_payoff_player1
  chosen_payoff_player2 = Exp1_payoff_player2

  print('player1_payoff:', chosen_payoff_player1)
  print('player2_payoff:', chosen_payoff_player2)
  input('ok, let\'s go?[any input will lead to running]:')

  def players_rewards():
    # ---- players config ----

    player1_start_cooperate_probability = game_start_probability(player1_start_theta)
    player2_start_cooperate_probability = game_start_probability(player2_start_theta)
    # assuming CC, CD, DC, DD

    player1_cooperate_probability = game_cooperate_probability(player1_game_theta)
    player2_cooperate_probability = game_cooperate_probability(player2_game_theta)
    reward_1, reward_2 = get_asymptotic_reward_mathematically(player1_start_cooperate_probability,
                                                              player2_start_cooperate_probability,
                                                              player1_cooperate_probability,
                                                              player2_cooperate_probability,
                                                              chosen_payoff_player1,
                                                              chosen_payoff_player2)
    return reward_1, reward_2


  # --- lola ---- #

  for i in range(10000000):
    V1, V2 = players_rewards()
    player1_start_cooperate_probability = game_start_probability(player1_start_theta)
    player2_start_cooperate_probability = game_start_probability(player2_start_theta)
    # assuming CC, CD, DC, DD
    player1_cooperate_probability = game_cooperate_probability(player1_game_theta)
    player2_cooperate_probability = game_cooperate_probability(player2_game_theta)

    if i % 1000 == 0:
      print('iter:', i, V1.item(), V2.item())
      p1cc, p1cd, p1dc, p1dd = player1_game_theta.tolist()
      p2cc, p2cd, p2dc, p2dd = player2_game_theta.tolist()
      print('p1CCl %.4f p1CDl  %.4f p1DCl %.4f p1DDl %.4f' % (p1cc, p1cd, p1dc, p1dd))
      print('p2CCl %.4f p2CDl  %.4f p2DCl %.4f p2DDl %.4f' % (p2cc, p2cd, p2dc, p2dd))

      p1cc, p1cd, p1dc, p1dd = player1_cooperate_probability.tolist()
      p2cc, p2cd, p2dc, p2dd = player2_cooperate_probability.tolist()
      print('p1CC %.4f p1CD  %.4f p1DC %.4f p1DD %.4f' % (p1cc, p1cd, p1dc, p1dd))
      print('p2CC %.4f p2CD  %.4f p2DC %.4f p2DD %.4f' % (p2cc, p2cd, p2dc, p2dd))
      print('p1_C %.4f p2_C %.4f' % (player1_start_cooperate_probability.tolist()[0],
                                     player2_start_cooperate_probability.tolist()[0]))

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
    dV2_wrt_player2_theta = torch.cat([dV2_wrt_player2_start_theta, dV2_wrt_player2_game_theta])
    dV1_wrt_player2_theta = torch.cat([dV1_wrt_player2_start_theta, dV1_wrt_player2_game_theta])
    dV1_wrt_player1_theta = torch.cat([dV1_wrt_player1_start_theta, dV1_wrt_player1_game_theta])
    dV2_wrt_player1_theta = torch.cat([dV2_wrt_player1_start_theta, dV2_wrt_player1_game_theta])

    V1_taylor = torch.dot(normalize(dV1_wrt_player2_theta), normalize(dV2_wrt_player2_theta))
    V2_taylor = torch.dot(normalize(dV2_wrt_player1_theta), normalize(dV1_wrt_player1_theta))

    if i % 1000 == 0:
      print(V1_taylor, V2_taylor)

    dV1_taylor_wrt_player1_start_theta, dV1_taylor_wrt_player1_game_theta = autograd.grad(
      outputs=V1_taylor,
      inputs=[player1_start_theta, player1_game_theta],
      retain_graph=True)

    dV2_taylor_wrt_player2_start_theta, dV2_taylor_wrt_player2_game_theta = autograd.grad(
      outputs=V2_taylor,
      inputs=[player2_start_theta, player2_game_theta],
      retain_graph=True)

    lr1 = 3e-2
    eta1 = 9e-2
    lr2 = 3e-2
    eta2 = 9e-2

    player1_start_theta.data.add_(dV1_wrt_player1_start_theta * lr1 + dV1_taylor_wrt_player1_start_theta * eta1)
    player1_game_theta.data.add_(dV1_wrt_player1_game_theta * lr1 + dV1_taylor_wrt_player1_game_theta * eta1)

    player2_start_theta.data.add_(dV2_wrt_player2_start_theta * lr2 + dV2_taylor_wrt_player2_start_theta * eta2)
    player2_game_theta.data.add_(dV2_wrt_player2_game_theta * lr2 + dV2_taylor_wrt_player2_game_theta * eta2)

    # plotting the probabilities
    if i % 1000 == 0:
      print('lr1 %.4f eta1 %.f4 lr2 %.4f eta2 %.4f' % (lr1, eta1, lr2, eta2))
      player1_start_cooperate_probability = game_start_probability(player1_start_theta)
      player2_start_cooperate_probability = game_start_probability(player2_start_theta)
      # assuming CC, CD, DC, DD

      player1_cooperate_probability = game_cooperate_probability(player1_game_theta)
      player2_cooperate_probability = game_cooperate_probability(player2_game_theta)

      plt.bar(['p1CC', 'p1CD', 'p1DC', 'p1DD'], player1_cooperate_probability.tolist())
      plt.bar(['p2CC', 'p2CD', 'p2DC', 'p2DD'], player2_cooperate_probability.tolist())
      plt.bar(['p1_C', 'p2_C'], [player1_start_cooperate_probability.tolist()[0],
                                 player2_start_cooperate_probability.tolist()[0]])
      plt.show()
      time.sleep(1)
