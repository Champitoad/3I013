from matplotlib import pyplot as plt
import numpy as np

x=[10,20,30,40,50,60,65,70,80,90,100]

"cursor : 8*8; 100 experiences, 100 episodes de 25 step pour le teste et 100 épisodes de 100 steps pour l'entrainement"
"entrainné pour une direction pour le ZERO"

y_true=[0.8778,0.906,0.929,0.9387,0.9487,0.9518,0.9522,0.9546,0.9557,0.9581,0.9588]
y_true2=[0.8684,0.9014,0.9228,0.9293,0.9388,0.9460,0.9531,0.9432,0.9422,0.9410,0.9425]
y_true3=(np.array(y_true2)+np.array(y_true))/2

#plt.plot(x,y_true)
#plt.plot(x,y_true2 )
bar_width=4
plt.xlabel('pourcentage de neurone par rapport à la couche précédente')
plt.ylabel('précision de prédiction')
plt.plot(x, y_true3)
#plt.bar(x,y_true3, bar_width)
plt.show()
#plt.savefig("experience_neurone_plot.png")
n=0.60
y=9422*2, 9509,9518
n=0.65
y=9551,9562

n=0.62
y=9529, 9559
y_true2=[0.8684,0.9014,0.9228,0.9293,0.9388,0.9460,0.9540,0.9551,0.9432,0.9422,0.9410,0.9425]
