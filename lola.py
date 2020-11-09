import random
import time

import torch
import torch.autograd as autograd
import matplotlib.pyplot as plt

from game import IPDGame, TwoPlayerSwitchGame


def game_cooperate_probability(game_theta: torch.tensor) -> torch.tensor:
  return torch.sigmoid(game_theta)


def game_start_probability(start_theta: torch.tensor) -> torch.tensor:
  return torch.sigmoid(start_theta)


def normalize(t: torch.tensor) -> torch.tensor:
  norm = torch.sqrt((t ** 2).sum())
  return t / norm


if __name__ == '__main__':

  player1_start_theta = torch.rand(1, requires_grad=True) - 0.5
  player1_game_theta = torch.rand(4, requires_grad=True) - 0.5
  player2_start_theta = torch.rand(1, requires_grad=True) - 0.5
  player2_game_theta = torch.rand(4, requires_grad=True) - 0.5

  # --- calculate the reward ---
  chosen_game = IPDGame()

  print('player1_payoff:', chosen_game.payoff_player1)
  print('player2_payoff:', chosen_game.payoff_player2)
  input('ok, let\'s go?[any input will lead to running]:')


  def players_rewards():
    # ---- players config ----
    player1_start_cooperate_probability = game_start_probability(player1_start_theta)
    player2_start_cooperate_probability = game_start_probability(player2_start_theta)
    # assuming CC, CD, DC, DD

    player1_cooperate_probability = game_cooperate_probability(player1_game_theta)
    player2_cooperate_probability = game_cooperate_probability(player2_game_theta)

    player1_params = {TwoPlayerSwitchGame.START_COOPERATION_PROBABILITY: player1_start_cooperate_probability,
                      TwoPlayerSwitchGame.GAME_COOPERATION_PROBABILITY: player1_cooperate_probability}
    player2_params = {TwoPlayerSwitchGame.START_COOPERATION_PROBABILITY: player2_start_cooperate_probability,
                      TwoPlayerSwitchGame.GAME_COOPERATION_PROBABILITY: player2_cooperate_probability}
    reward_1, reward_2 = chosen_game.compute_reward([player1_params, player2_params])

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

    # V1_taylor = torch.dot(normalize(dV1_wrt_player2_theta), normalize(dV2_wrt_player2_theta))
    # V2_taylor = torch.dot(normalize(dV2_wrt_player1_theta), normalize(dV1_wrt_player1_theta))

    V1_taylor = torch.dot(dV1_wrt_player2_theta.detach(), dV2_wrt_player2_theta)
    V2_taylor = torch.dot(dV2_wrt_player1_theta.detach(), dV1_wrt_player1_theta)

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
    eta1 = 3e-1
    lr2 = 3e-2
    eta2 = 3e-1

    # if i < 5000:
    #   eta1 = 0
    #   eta2 = 0

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
