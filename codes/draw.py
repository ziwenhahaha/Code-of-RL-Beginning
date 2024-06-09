import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
def draw(state_value, policy ):
    x = np.linspace(0,5,5)
    y = np.linspace(0,5,5)
    
    X, Y = np.meshgrid(x,y)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection = '3d')
    ax1.plot_surface(Y,X,state_value, cmap='summer')
    ax1.set_title('3D Plot')

    ax2 = fig.add_subplot(122)
    im = ax2.imshow(state_value, cmap='summer',origin='upper')
    ax2.set_title('2D Heatmap')

    for i in range(len(X)):
        for j in range(len(Y)):
            text = ax2.text(j,i, round(state_value[i][j], 2),ha='center',va='center',color='black',alpha=1,fontsize=8)

    logos = ['↑','→','↓','←','○']

    for i in range(len(X)):
        for j in range(len(Y)):
            text = ax2.text(j,i, logos[policy[i*5+j]],ha='center',va='center',color='pink',alpha=0.6,fontsize=30)
    plt.colorbar(im)
    plt.show()