import random
import time

import torch
import torch.autograd as autograd
import matplotlib.pyplot as plt

from agent import LOLAAgent
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

  player1 = LOLAAgent({'start_theta': player1_start_theta,
                       'game_theta': player1_game_theta}, lr=3e-2, eta=9e-1)
  player2 = LOLAAgent({'start_theta': player2_start_theta,
                       'game_theta': player2_game_theta}, lr=3e-2, eta=9e-1)

  def players_rewards():
    # ---- players config ----
    player1_start_cooperate_probability = game_start_probability(player1.parameters['start_theta'])
    player2_start_cooperate_probability = game_start_probability(player2.parameters['start_theta'])
    # assuming CC, CD, DC, DD

    player1_cooperate_probability = game_cooperate_probability(player1.parameters['game_theta'])
    player2_cooperate_probability = game_cooperate_probability(player2.parameters['game_theta'])

    player1_params = {TwoPlayerSwitchGame.START_COOPERATION_PROBABILITY: player1_start_cooperate_probability,
                      TwoPlayerSwitchGame.GAME_COOPERATION_PROBABILITY: player1_cooperate_probability}
    player2_params = {TwoPlayerSwitchGame.START_COOPERATION_PROBABILITY: player2_start_cooperate_probability,
                      TwoPlayerSwitchGame.GAME_COOPERATION_PROBABILITY: player2_cooperate_probability}
    reward_1, reward_2 = chosen_game.compute_reward([player1_params, player2_params])

    return reward_1, reward_2


  # --- lola ---- #
  for i in range(10000000):
    V1, V2 = players_rewards()
    player1.update_parameters(V1, V2, player2)
    player2.update_parameters(V2, V1, player1)

    # plotting the probabilities
    if i % 1000 == 0:
      player1_start_cooperate_probability = game_start_probability(player1.parameters['start_theta'])
      player2_start_cooperate_probability = game_start_probability(player2.parameters['start_theta'])
      # assuming CC, CD, DC, DD

      player1_cooperate_probability = game_cooperate_probability(player1.parameters['game_theta'])
      player2_cooperate_probability = game_cooperate_probability(player2.parameters['game_theta'])

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

      print('lr1 %.4f eta1 %.f4 lr2 %.4f eta2 %.4f' % (player1.lr, player1.eta, player2.lr, player2.eta))
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
