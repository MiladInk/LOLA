import random
from typing import Dict, List

import torch


class GameParamsShape:
  def __init__(self, parameter_name_to_shape: Dict[str, int]):
    self.parameter_name_to_shape = parameter_name_to_shape


class Game:
  def __init__(self, game_params: GameParamsShape):
    self.game_params = game_params

  def compute_reward(self, agents_params: List[Dict[str, torch.tensor]]) -> List[torch.tensor]:
    raise NotImplementedError


class TwoPlayerSwitchGame(Game):
  START_COOPERATION_PROBABILITY = 'start_cooperation_probability'
  GAME_COOPERATION_PROBABILITY = 'game_cooperation_probability'

  def __init__(self, payoff_player1: torch.tensor, payoff_player2: torch.tensor):
    self.payoff_player1 = payoff_player1
    self.payoff_player2 = payoff_player2
    game_params_shape = GameParamsShape({
      self.START_COOPERATION_PROBABILITY: 1,
      self.GAME_COOPERATION_PROBABILITY: 4,
    })
    super(TwoPlayerSwitchGame, self).__init__(game_params_shape)

  def compute_reward(self, player_params: List[Dict[str, torch.tensor]]) -> List[torch.tensor]:
    player1_params = player_params[0]
    player2_params = player_params[1]
    player1_start_cooperate_probability = player1_params[self.START_COOPERATION_PROBABILITY]
    player2_start_cooperate_probability = player2_params[self.START_COOPERATION_PROBABILITY]
    player1_cooperate_probability = player1_params[self.GAME_COOPERATION_PROBABILITY]
    p2CC = player2_params[self.GAME_COOPERATION_PROBABILITY][0]
    p2CD = player2_params[self.GAME_COOPERATION_PROBABILITY][1]
    p2DC = player2_params[self.GAME_COOPERATION_PROBABILITY][2]
    p2DD = player2_params[self.GAME_COOPERATION_PROBABILITY][3]
    player2_cooperate_probability = torch.stack([p2CC, p2DC, p2CD, p2DD])

    reward_1, reward_2 = get_asymptotic_reward_mathematically(player1_start_cooperate_probability,
                                                              player2_start_cooperate_probability,
                                                              player1_cooperate_probability,
                                                              player2_cooperate_probability,
                                                              self.payoff_player1,
                                                              self.payoff_player2)
    return [reward_1, reward_2]


class IPDGame(TwoPlayerSwitchGame):
  def __init__(self):
    IPD_payoff_player1 = torch.tensor([[-1, -3],
                                       [0, -2]])

    IPD_payoff_player2 = torch.tensor([[-1, 0],
                                       [-3, -2]])

    super(IPDGame, self).__init__(IPD_payoff_player1, IPD_payoff_player2)


class ChickenGame(TwoPlayerSwitchGame):
  def __init__(self):
    Chicken_payoff_player1 = torch.tensor([[0, -1],
                                           [10, -200]])

    Chicken_payoff_player2 = torch.tensor([[0, 10],
                                           [-1, -200]])

    super(ChickenGame, self).__init__(Chicken_payoff_player1, Chicken_payoff_player2)


class MatchingPenniesGame(TwoPlayerSwitchGame):
  def __init__(self):
    MatchingPennies_payoff_player1 = torch.tensor([[1, 0],
                                                   [0, 1]])

    MatchingPennies_payoff_player2 = torch.tensor([[0, 1],
                                                   [1, 0]])
    super(MatchingPenniesGame, self).__init__(MatchingPennies_payoff_player1, MatchingPennies_payoff_player2)


class ExpGame(TwoPlayerSwitchGame):
  def __init__(self):
    Exp1_payoff_player1 = torch.tensor([[0, 2.5],
                                        [-1., 1]])

    Exp1_payoff_player2 = torch.tensor([[0, -1],
                                        [2.5, 1]])
    super(ExpGame, self).__init__(Exp1_payoff_player1, Exp1_payoff_player2)


class RandomGame(TwoPlayerSwitchGame):
  def __init__(self):
    rand_payoff_player1, rand_payoff_player2 = self.get_random_payoff_matrix()
    super(RandomGame, self).__init__(rand_payoff_player1, rand_payoff_player2)

  @staticmethod
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


alpha = 0.96


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


# noinspection DuplicatedCode
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


