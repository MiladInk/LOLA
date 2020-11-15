import argparse
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

from agent import LOLAAgent, NormLOLAAgent, Agent
from game import IPDGame, TwoPlayerSwitchGame
from multiprocessing import Pool


def get_game_cooperate_probability(game_theta: torch.tensor) -> torch.tensor:
  return torch.sigmoid(game_theta)


def get_game_start_probability(start_theta: torch.tensor) -> torch.tensor:
  return torch.sigmoid(start_theta)


def normalize(t: torch.tensor) -> torch.tensor:
  norm = torch.sqrt((t ** 2).sum())
  return t / norm


def players_rewards(player1, player2, game):
  # ---- players config ----
  player1_start_cooperate_probability = get_game_start_probability(player1.parameters['start_theta'])
  player2_start_cooperate_probability = get_game_start_probability(player2.parameters['start_theta'])
  # assuming CC, CD, DC, DD

  player1_cooperate_probability = get_game_cooperate_probability(player1.parameters['game_theta'])
  player2_cooperate_probability = get_game_cooperate_probability(player2.parameters['game_theta'])

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
    self.v1s = []
    self.p1policies = []
    self.v2s = []
    self.p2policies = []


def get_player_policy(player: Agent):
  start_cooperate_probability = get_game_start_probability(player.parameters['start_theta']).detach()
  game_cooperate_probability = get_game_cooperate_probability(player.parameters['game_theta']).detach()

  policy = {TwoPlayerSwitchGame.START_COOPERATION_PROBABILITY: start_cooperate_probability,
            TwoPlayerSwitchGame.GAME_COOPERATION_PROBABILITY: game_cooperate_probability}

  return policy


def log_players_policies(player1: Agent, player2: Agent, run_log: RunLog):
  p1policy = get_player_policy(player1)
  p2policy = get_player_policy(player2)
  run_log.p1policies.append(p1policy)
  run_log.p2policies.append(p2policy)


def train_lola_and_norm_lola(chosen_game, iterations_to_train: int):
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

  lola_players_run_log = RunLog()
  norm_lola_players_run_log = RunLog()
  # --- lola ---- #
  for iteration in range(iterations_to_train):  # TODO a more disciplined way to end the training
    lola_v1, lola_v2 = update_these_two(lola_player1, lola_player2, chosen_game)
    lola_players_run_log.v1s.append(lola_v1.item())
    lola_players_run_log.v2s.append(lola_v2.item())

    norm_lola_v1, norm_lola_v2 = update_these_two(norm_lola_player1, norm_lola_player2, chosen_game)
    norm_lola_players_run_log.v1s.append(norm_lola_v1.item())
    norm_lola_players_run_log.v2s.append(norm_lola_v2.item())

    if iteration % 1000 == 0:
      print('LOLA %d : v1 %.4f v2 %.4f NORMLOLA %.4f %.4f' % (iteration, lola_v1, lola_v2, norm_lola_v1, norm_lola_v2))

  log_players_policies(lola_player1, lola_player2, lola_players_run_log)
  log_players_policies(norm_lola_player1, norm_lola_player2, norm_lola_players_run_log)

  return lola_players_run_log, norm_lola_players_run_log


def is_policy_tit_for_tat(policy: Dict[str, torch.tensor], threshold: float) -> bool:
  start_cooperation_probability = policy[TwoPlayerSwitchGame.START_COOPERATION_PROBABILITY]
  game_cooperation_probability = policy[TwoPlayerSwitchGame.GAME_COOPERATION_PROBABILITY]
  pCC = game_cooperation_probability[0]
  pCD = game_cooperation_probability[1]
  pDC = game_cooperation_probability[2]
  pDD = game_cooperation_probability[3]
  if pCC < (1.-threshold):
    return False
  if pCD > 0.+threshold:
    return False
  if pDC < (1.-threshold):
    return False
  if pDD > 0.+threshold:
    return False

  return True


def get_tit_for_tat_percent_in_policies(policies: List[Dict[str, torch.tensor]], threshold: float) -> float:
  are_policies_tit_for_tat = []
  for policy in policies:
    are_policies_tit_for_tat.append(is_policy_tit_for_tat(policy, threshold=threshold))

  tit_for_tat_count = 0
  for flag in are_policies_tit_for_tat:
    if flag:
      tit_for_tat_count += 1

  return tit_for_tat_count/len(policies)


def plot_land():
  results = torch.load('saved_results.pickle')

  lola_final_vs = []
  norm_lola_final_vs = []
  lola_final_policies = []
  lola_ith_policies = defaultdict(list)
  norm_lola_final_policies = []
  norm_lola_ith_policies = defaultdict(list)

  for result in results:
    lola_players_log, norm_lola_players_log = result
    lola_final_vs += [lola_players_log.v1s[-1], lola_players_log.v2s[-1]]
    norm_lola_final_vs += [norm_lola_players_log.v1s[-1], norm_lola_players_log.v2s[-1]]
    lola_final_policies += [lola_players_log.p1policies[-1], lola_players_log.p2policies[-1]]
    norm_lola_final_policies += [norm_lola_players_log.p1policies[-1], norm_lola_players_log.p2policies[-1]]

    for i in range(len(lola_players_log.p1policies)):
      lola_ith_policies[i] += [lola_players_log.p1policies[i], lola_players_log.p2policies[i]]
      norm_lola_ith_policies[i] += [norm_lola_players_log.p1policies[i], norm_lola_players_log.p2policies[i]]

  print(get_tit_for_tat_percent_in_policies(lola_final_policies, threshold=0.5))

  fig = plt.figure(figsize=(24, 8))
  ax1 = fig.add_subplot(3, 1, 1)
  ax1.hist(lola_final_vs, range=(-2., -1.))
  ax1.set_xlabel('final lola agent value')

  ax2 = fig.add_subplot(3, 1, 2)
  ax2.hist(norm_lola_final_vs, range=(-2., -1.))
  ax2.set_xlabel('final norm-lola agent value')

  ax3 = fig.add_subplot(3, 1, 3)
  thresholds = np.arange(0., 1., step=0.01)
  lola_final_policies_tit_for_tat_percent = []
  norm_lola_final_policies_tit_for_tat_percent = []
  for threshold in thresholds:
    lola_final_policies_tit_for_tat_percent.append(get_tit_for_tat_percent_in_policies(lola_final_policies,
                                                                                       threshold))

    norm_lola_final_policies_tit_for_tat_percent.append(get_tit_for_tat_percent_in_policies(norm_lola_final_policies,
                                                                                            threshold))

  ax3.plot(lola_final_policies_tit_for_tat_percent, thresholds)
  ax3.plot(norm_lola_final_policies_tit_for_tat_percent, thresholds)
  ax3.legend(['LOLA-LOLA', 'NORMLOLA-NORMLOLA'])
  ax3.set_xlabel('threshold')
  ax3.set_ylabel('tit-for-tat-percent')

  fig.tight_layout()
  plt.savefig('lola_vs_norm_lola.png')


if __name__ == '__main__':
  # --- calculate the reward ---
  parser = argparse.ArgumentParser()
  parser.add_argument("--runs", help="number of the runs to test", default=100, type=int)
  parser.add_argument("--iterations", help="number of iterations to train each agent", default=20000, type=int)
  args = parser.parse_args()
  print('%d runs to test' % args.runs)

  selected_game = IPDGame()

  print('player1_payoff:', selected_game.payoff_player1)
  print('player2_payoff:', selected_game.payoff_player2)

  p = Pool(8)
  results = p.starmap(train_lola_and_norm_lola, [(selected_game, args.iterations)] * args.runs)
  torch.save(results, 'saved_results.pickle')
  plot_land()
