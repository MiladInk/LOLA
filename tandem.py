import time

import numpy as np
import torch
import torch.autograd as autograd
import matplotlib.pyplot as plt


def normalize(t: torch.tensor) -> torch.tensor:
  norm = torch.sqrt((t**2).sum())
  return t/norm


if __name__ == '__main__':

  player1_theta = torch.rand(1, requires_grad=True) - 0.5
  player2_theta = torch.rand(1, requires_grad=True) - 0.5


  def players_rewards():
    x = player1_theta
    y = player2_theta
    loss_1 = (x+y)**2-2*x
    loss_2 = (x+y)**2-2*y
    return -loss_1, -loss_2


  v1s = []
  v2s = []
  v1_taylors = []
  v2_taylors = []

  for i in range(100000000):
    V1, V2 = players_rewards()

    v1s.append(V1.item())
    v2s.append(V2.item())

    if i % 1 == 0:
      print('iter: %d %.4f %.4f' % (i, V1.item(), V2.item()))
      print('x value: %.4f y value: %.4f' % (player1_theta, player2_theta))

    dV1_wrt_player1_theta = autograd.grad(
      outputs=V1,
      inputs=[player1_theta],
      create_graph=True,
      retain_graph=True)[0]

    # print(dV1_wrt_player1_theta)

    dV2_wrt_player2_theta = autograd.grad(
      outputs=V2,
      inputs=[player2_theta],
      create_graph=True,
      retain_graph=True)[0]

    dV1_wrt_player2_theta = autograd.grad(
      outputs=V1,
      inputs=[player2_theta],
      create_graph=True,
      retain_graph=True)[0]

    dV2_wrt_player1_theta = autograd.grad(
      outputs=V2,
      inputs=[player1_theta],
      create_graph=True,
      retain_graph=True)[0]

    # --- LOLA ----
    V1_taylor = torch.dot(dV2_wrt_player2_theta, dV1_wrt_player2_theta.detach())
    V2_taylor = torch.dot(dV1_wrt_player1_theta,dV2_wrt_player1_theta.detach())

    # ---- mLOLA ----
    # V1_taylor = torch.dot(normalize(dV2_wrt_player2_theta), normalize(dV1_wrt_player2_theta))
    # V2_taylor = torch.dot(normalize(dV1_wrt_player1_theta), normalize(dV2_wrt_player1_theta))

    v1_taylors.append(V1_taylor.item())
    v2_taylors.append(V2_taylor.item())

    print('V1_taylor %.4f V2_taylor %.4f' % (V1_taylor, V2_taylor))

    dV1_taylor_wrt_player1_theta = autograd.grad(
      outputs=V1_taylor,
      inputs=[player1_theta],
      retain_graph=True)[0]

    dV2_taylor_wrt_player2_theta = autograd.grad(
      outputs=V2_taylor,
      inputs=[player2_theta],
      retain_graph=True)[0]

    lr1 = 3e-2
    eta1 = 3e-1
    lr2 = 3e-2
    eta2 = 3e-1

    player1_theta.data.add_(dV1_wrt_player1_theta * lr1 + dV1_taylor_wrt_player1_theta * eta1)
    player2_theta.data.add_(dV2_wrt_player2_theta * lr2 + dV2_taylor_wrt_player2_theta * eta2)

    if i % 20 == 0:
      plt.figure(figsize=(15, 8))
      plt.subplot(2, 1, 1)
      plt.plot(np.arange(len(v1s)), v1s)
      plt.plot(np.arange(len(v1s)), v2s)
      plt.legend(['V1', 'V2'], loc='upper left', frameon=True, framealpha=1, ncol=3)
      plt.subplot(2, 1, 2)
      plt.plot(np.arange(len(v1s)), v1_taylors)
      plt.plot(np.arange(len(v1s)), v2_taylors)
      plt.legend(['V1_taylor_LOLA_term', 'V2_taylor_LOLA_term'], loc='upper left', frameon=True, framealpha=1, ncol=3)
      plt.show()

      time.sleep(1)