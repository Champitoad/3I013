from matplotlib import pyplot as plt
import numpy as np

x=[10,20,30,40,50,60,62,65,70,80,90,100]

"cursor : 8*8; 100 experiences, 100 episodes de 25 step pour le teste et 100 épisodes de 100 steps pour l'entrainement"
"entrainné pour une direction pour le ZERO"

y=[0.902,0.870,0.907, 0.885,0.916,0.853,0.855,0.873,0.804,0.863]
y2=[0.905,0.890,0.865,0.863,0.880,0.926,0.892,0.844,0.888,0.879]
y3=[0.884,0.846,0.846,0.870,0.877,0.877,0.912,0.816,0.884,0.927]
y4=[0.889,0.862,0.830,0.870,0.862,0.910,0.826,0.885,0.880,0.842]
y_true=[0.8543,0.8625,0.8662,0.8706,0.8703,0.8802,0.8874,0.8862,0.8781,0.8804]
y_true2=[0.8684,0.9014,0.9228,0.9293,0.9388,0.9460,0.9540,0.9551,0.9432,0.9422,0.9410,0.9425]

#plt.plot(x,y_true)
plt.plot(x,y_true2, )
plt.show()
#plt.savefig("experience_neurone_plot.png")
n=0.60
y=9422*2, 9509
n=0.65
y=9551,9562

n=0.62
y=9529, 9559