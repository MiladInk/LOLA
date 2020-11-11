import matplotlib.pyplot as plt
import torch

from agent import LOLAAgent, NormLOLAAgent, Agent
from game import IPDGame, TwoPlayerSwitchGame
from multiprocessing import Pool


def game_cooperate_probability(game_theta: torch.tensor) -> torch.tensor:
  return torch.sigmoid(game_theta)


def game_start_probability(start_theta: torch.tensor) -> torch.tensor:
  return torch.sigmoid(start_theta)


def normalize(t: torch.tensor) -> torch.tensor:
  norm = torch.sqrt((t ** 2).sum())
  return t / norm


def players_rewards(player1, player2, game):
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
  reward_1, reward_2 = game.compute_reward([player1_params, player2_params])

  return reward_1, reward_2


def update_these_two(player1, player2, game):
  V1, V2 = players_rewards(player1, player2, game)
  player1.update_parameters(V1, V2, player2)
  player2.update_parameters(V2, V1, player1)
  return V1, V2


class RunLog:
  def __init__(self):
    self.vs = []
    self.final_start_cooperate_probability = None
    self.final_game_cooperate_probability = None


def log_policy(player: Agent, run_log: RunLog):
  run_log.final_start_cooperate_probability = game_start_probability(player.parameters['start_theta']).detach()
  run_log.final_game_cooperate_probability = game_cooperate_probability(player.parameters['game_theta']).detach()


def train_lola_and_norm_lola(chosen_game):
  torch.seed()
  player1_start_theta = torch.rand(1) * 8 - 4
  player1_game_theta = torch.rand(4) * 8 - 4
  player2_start_theta = torch.rand(1) * 8 - 4
  player2_game_theta = torch.rand(4) * 8 - 4

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

  lola_player1_run_log = RunLog()
  lola_player2_run_log = RunLog()
  norm_lola_player1_run_log = RunLog()
  norm_lola_player2_run_log = RunLog()
  # --- lola ---- #
  for epoch in range(20000):  # TODO a more disciplined way to end the training
    lola_v1, lola_v2 = update_these_two(lola_player1, lola_player2, chosen_game)
    lola_player1_run_log.vs.append(lola_v1.item())
    lola_player2_run_log.vs.append(lola_v2.item())

    norm_lola_v1, norm_lola_v2 = update_these_two(norm_lola_player1, norm_lola_player2, chosen_game)
    norm_lola_player1_run_log.vs.append(norm_lola_v1.item())
    norm_lola_player2_run_log.vs.append(norm_lola_v2.item())

    if epoch % 1000 == 0:
      print('LOLA: v1 %.4f v2 %.4f NORMLOLA %.4f %.4f' % (lola_v1, lola_v2, norm_lola_v1, norm_lola_v2))

  log_policy(lola_player1, lola_player1_run_log)
  log_policy(lola_player2, lola_player2_run_log)
  log_policy(norm_lola_player1, norm_lola_player1_run_log)
  log_policy(norm_lola_player2, norm_lola_player2_run_log)

  return lola_player1_run_log, lola_player2_run_log, norm_lola_player1_run_log, norm_lola_player2_run_log


if __name__ == '__main__':
  # --- calculate the reward ---
  selected_game = IPDGame()

  print('player1_payoff:', selected_game.payoff_player1)
  print('player2_payoff:', selected_game.payoff_player2)
  input('ok, let\'s go?[any input will lead to running]:')
  lola_final_vs = []
  norm_lola_final_vs = []
  p = Pool(8)
  results = p.map(train_lola_and_norm_lola, [selected_game] * 100)
  print(results)

  for result in results:
    lola_player1_log, lola_player2_log, norm_lola_player1_log, norm_lola_player2_log = result
    lola_final_vs += [lola_player1_log.vs[-1], lola_player2_log.vs[-1]]
    norm_lola_final_vs += [norm_lola_player1_log.vs[-1], norm_lola_player2_log.vs[-1]]

  plt.figure(figsize=(16, 8))
  plt.subplot(2, 1, 1)
  plt.hist(lola_final_vs, range=(-2., -1.))
  plt.xlabel('final lola agent value')
  plt.subplot(2, 1, 2)
  plt.hist(norm_lola_final_vs, range=(-2., -1.))
  plt.xlabel('final norm lola agent value')
  plt.savefig('lola_vs_norm_lola.png')
