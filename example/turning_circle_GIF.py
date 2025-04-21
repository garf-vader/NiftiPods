import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.widgets import Slider

# create a figure and axes
fig = plt.figure(figsize=(12,6))
fig.suptitle('Turning Circle At 10mph for various Banking Angles')
ax1 = plt.subplot(1,2,1)   
ax2 = plt.subplot(1,2,2)
fig.subplots_adjust(bottom=0.25)
# set up the subplots as needed
ax1.set_xlim(( -2, 2))            
ax1.set_ylim((-2, 2))

ax2.set_xlim((-10,10))
ax2.set_ylim((-10,10))
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

txt_title1 = ax1.set_title('')
txt_title2 = ax2.set_title("")


rect = patches.Rectangle((-0.5, -0.5), 1, 1, angle = 0, rotation_point='center', linewidth=1, edgecolor='r', facecolor='none')
ax1.add_patch(rect)

circ = patches.Circle((0, 0), 1, linewidth=1, edgecolor='r', facecolor='none')
ax2.add_patch(circ)


def drawframe(n):
    bank = n+1
    v_kph = 16
    v = v_kph/3.6
    r = v**2 / (9.81 * np.tan(np.deg2rad(bank)))

    rect.angle = bank
    circ.radius = r

    txt_title1.set_text('Bank Angle = {0:4d}$^\circ$'.format(n))
    txt_title2.set_text('Radius = {0:2f}m'.format(r))


from matplotlib import animation

anim = animation.FuncAnimation(fig, drawframe, frames=60, interval=100, blit=False )

# 1. render and display the desired animation by HTML
from IPython.display import HTML
HTML(anim.to_html5_video())
# 2. render and display the desired animation by rc
from matplotlib import rc
# equivalent to rcParams['animation.html'] = 'html5'
rc('animation', html='html5')
anim

from matplotlib.animation import PillowWriter
anim.save("turning_circ.gif", dpi=250, writer=PillowWriter(fps=4))