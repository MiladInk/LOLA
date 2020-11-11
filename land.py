import torch

from agent import LOLAAgent, NormLOLAAgent, PointNormLOLAAgent
from game import IPDGame, TwoPlayerSwitchGame, ChickenGame
import numpy as np
import matplotlib.pyplot as plt


def game_cooperate_probability(game_theta: torch.tensor) -> torch.tensor:
  return torch.sigmoid(game_theta)


def game_start_probability(start_theta: torch.tensor) -> torch.tensor:
  return torch.sigmoid(start_theta)


def normalize(t: torch.tensor) -> torch.tensor:
  norm = torch.sqrt((t ** 2).sum())
  return t / norm


if __name__ == '__main__':
  population_size = 2
  player_thetas = []
  player1_start_theta = torch.rand(1)*8 - 4
  player1_game_theta = torch.rand(4)*8 - 4
  player2_start_theta = torch.rand(1)*8 - 4
  player2_game_theta = torch.rand(4)*8 - 4

  lola_player1_theta = {'start_theta': player1_start_theta.detach().clone().requires_grad_(),
                        'game_theta': player1_game_theta.detach().clone().requires_grad_()}

  lola_player2_theta = {'start_theta': player2_start_theta.detach().clone().requires_grad_(),
                        'game_theta': player2_game_theta.detach().clone().requires_grad_()}

  norm_lola_player1_theta = {'start_theta': player1_start_theta.detach().clone().requires_grad_(),
                             'game_theta': player1_game_theta.detach().clone().requires_grad_()}

  norm_lola_player2_theta = {'start_theta': player2_start_theta.detach().clone().requires_grad_(),
                             'game_theta': player2_game_theta.detach().clone().requires_grad_()}

  lola_player1 = LOLAAgent(lola_player1_theta, lr=3e-2, eta=3e-1)
  lola_player2 = LOLAAgent(lola_player2_theta, lr=3e-2, eta=3e-1)
  norm_lola_player1 = NormLOLAAgent(norm_lola_player1_theta, lr=3e-2, eta=9e-1)
  norm_lola_player2 = NormLOLAAgent(norm_lola_player2_theta, lr=3e-2, eta=9e-1)

  # --- calculate the reward ---
  chosen_game = IPDGame()

  print('player1_payoff:', chosen_game.payoff_player1)
  print('player2_payoff:', chosen_game.payoff_player2)
  input('ok, let\'s go?[any input will lead to running]:')

  def players_rewards(player1, player2):
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

  def update_these_two(player1, player2):
    V1, V2 = players_rewards(player1, player2)
    player1.update_parameters(V1, V2, player2)
    player2.update_parameters(V2, V1, player1)
    return V1, V2

  # --- lola ---- #
  for epoch in range(10000000):
    lola_v1, lola_v2 = update_these_two(lola_player1, lola_player2)
    norm_lola_v1, norm_lola_v2 = update_these_two(norm_lola_player1, norm_lola_player2)
    if epoch % 1000 == 0:
      print('LOLA: v1 %.4f v2 %.4f NORMLOLA %.4f %.4f' % (lola_v1, lola_v2, norm_lola_v1, norm_lola_v2))


