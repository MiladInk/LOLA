import torch
import torch.autograd as autograd
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
  normalizing_factor = 1 / (1-alpha)
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

  player1_start_theta = torch.zeros(1, requires_grad=True)
  player2_start_theta = torch.ones(1, requires_grad=True)
  player1_game_theta = torch.ones(4, requires_grad=True)
  player2_game_theta = torch.ones(4, requires_grad=True)

  # --- calculate the reward ---
  def result():

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
                                                              IPD_payoff_player1,
                                                              IPD_payoff_player2)
    return reward_1, reward_2


  for _ in range(100000):
    reward1, reward2 = result()
    print(reward1.item(), reward2.item())

    res = autograd.grad(outputs=reward1, inputs=[player1_start_theta, player1_game_theta], retain_graph=True)
    res2 = autograd.grad(outputs=reward2, inputs=[player2_start_theta, player2_game_theta])
    lr = 3e-2

    player1_start_theta.data.add_(lr * res[0])
    player1_game_theta.data.add_(lr * res[1])
    player2_start_theta.data.add_(lr * res2[0])
    player2_game_theta.data.add_(lr * res2[1])












