# Plot the DIAM greening function

#-----------------------------------------------------------------------

# MODULES

import numpy as np
import matplotlib.pyplot as plt

from module_coupling import get_chit

#-----------------------------------------------------------------------

# CALCULATIONS

chit = get_chit()
greening = 1 - chit

years = np.arange(1990, 1990+chit.shape[0], 1)

#-----------------------------------------------------------------------

# PLOT

cm = 1/2.54  # centimeters in inches

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6*cm, 4.5*cm))

ax.plot(years, greening, color='g', linewidth=1)
ax.set_ylabel('Fraction of green energy', fontsize=6)
ax.set_xlabel(r'Year', fontsize=6)
ax.xaxis.set_tick_params(labelsize=5)
ax.yaxis.set_tick_params(labelsize=5)

fig.subplots_adjust(bottom=0.2, left=0.18, top=0.95, right=0.95)

fig.savefig('figures/figure_greening_function.pdf')

#--------------------------------------------------------------------
