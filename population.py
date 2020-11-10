import torch

from agent import LOLAAgent
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
  population_size = 10
  player_thetas = []
  for i in range(population_size):
    player_start_theta = torch.rand(1, requires_grad=True) - 0.5
    player_game_theta = torch.rand(4, requires_grad=True) - 0.5
    player_thetas.append(
      {'start_theta': player_start_theta,
       'game_theta': player_game_theta}
    )
  players = []
  for i in range(population_size):
    players.append(LOLAAgent(player_thetas[i], lr=3e-2, eta=3e-2))

  # --- calculate the reward ---
  chosen_game = ChickenGame()

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
  V_epoch_list = []
  for epoch in range(10000000):
    epoch_v_list = []
    for p1 in range(population_size):
      for p2 in range(population_size):
        if p1 != p2:
          v1, v2 = update_these_two(players[p1], players[p2])
          epoch_v_list += [v1.item(), v2.item()]

    V_epoch_list.append(sum(epoch_v_list)/len(epoch_v_list))
    epoch_v_list = []
    print('iter %d - avg_value: %.4f' % (epoch, V_epoch_list[-1]))
    if epoch % 500 == 0 and epoch > 0:
      plt.figure(figsize=(15, 8))
      plt.plot(np.arange(len(V_epoch_list)), V_epoch_list)
      plt.title('Iterated IPD with 10 LOLA')
      plt.xlabel('Epoch')
      plt.ylabel('Average Loss')
      plt.show()

      plt.figure(figsize=(12, 16))
      idx = 1
      for i in range(population_size):
        plt.subplot(3, 4, idx)
        player_start_cooperate_probability = game_start_probability(players[i].parameters['start_theta'])
        player_cooperate_probability = game_cooperate_probability(players[i].parameters['game_theta'])
        plt.bar(['pCC', 'pCD', 'pDC', 'pDD'], player_cooperate_probability.tolist())
        plt.bar(['pC'], [player_start_cooperate_probability.tolist()[0]])
        idx += 1

      plt.title('learned_policies')
      plt.show()


