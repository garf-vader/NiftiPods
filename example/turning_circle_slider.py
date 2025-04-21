import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.widgets import Slider

# create a figure and axes
fig = plt.figure(figsize=(12,6.6))
fig.suptitle('Turning Circle At 10mph for various Banking Angles')
ax1 = plt.subplot(1,2,1)   
ax2 = plt.subplot(1,2,2)
fig.subplots_adjust(bottom=0.25)
ax1.set_xlim(( -2, 2))            
ax1.set_ylim((-2, 2))

ax2.set_xlim((-10,10))
ax2.set_ylim((-10,10))

txt_title1 = ax1.set_title("")
txt_title2 = ax2.set_title("")

rect = patches.Rectangle((-0.5, -0.5), 1, 1, angle = 0, rotation_point='center', linewidth=1, edgecolor='r', facecolor='none')
ax1.add_patch(rect)

circ = patches.Circle((0, 0), 1, linewidth=1, edgecolor='r', facecolor='none')
ax2.add_patch(circ)

bank = 1
pod_label = ax1.text(0, 0, "NIFTI", ha="center", va="center", rotation=bank, size=15)

v_kph = 16
v = v_kph/3.6
r = v**2 / (9.81 * np.tan(np.deg2rad(bank)))

rect.angle = bank
circ.radius = r

txt_title1.set_text('Bank Angle = {0:4d}$^\circ$'.format(bank))
txt_title2.set_text('Radius = {0:2f}m'.format(r))

axbank = fig.add_axes([0.25, 0.1, 0.65, 0.03])
bank_slider = Slider(
    ax=axbank,
    label='Bank Angle [$^\circ$]',
    valmin=1,
    valmax=60,
    valinit=bank,
)

def update(val):
    bank = bank_slider.val

    pod_label.set_rotation(bank)

    v_kph = 16
    v = v_kph/3.6
    r = v**2 / (9.81 * np.tan(np.deg2rad(bank)))

    rect.angle = bank
    circ.radius = r

    txt_title1.set_text('Bank Angle = {0:2f}$^\circ$'.format(bank))
    txt_title2.set_text('Radius = {0:2f}m'.format(r))

    #fig.canvas.draw_idle()
    fig.canvas.draw()

bank_slider.on_changed(update)

plt.show()