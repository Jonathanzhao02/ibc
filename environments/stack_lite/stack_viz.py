import numpy as np
import matplotlib.pyplot as plt

def visualize(pos_log):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  for obj in pos_log[0].keys():
    pos = np.array([log[obj] for log in pos_log]).T
    ax.plot(pos[0], pos[1], pos[2], label=obj)
  
  ax.legend()
  return fig, ax
