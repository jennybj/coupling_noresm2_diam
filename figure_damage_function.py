# Plot the DIAM damage function

#-----------------------------------------------------------------------

# MODULES

import numpy as np
import matplotlib.pyplot as plt

from module_coupling import damage_function

#-----------------------------------------------------------------------

# PARAMETERS

temp_opt = 12.609  # degC
kappa_plus = 0.00327721
kappa_minus = 0.00362887
alpha = 0.36  # capital’s share of income (capital share + labor share = 1)


#-----------------------------------------------------------------------

# CALCULATIONS

temp = np.arange(-40, 40, 0.1)  # max and min from NorESM approx -55 to 35

shape = damage_function(temp, temp_opt, kappa_plus,
                                    kappa_minus)


#-----------------------------------------------------------------------

# PLOT

cm = 1/2.54  # centimeters in inches

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6*cm, 4.5*cm))

ax.plot(temp, shape, color='r', linewidth=1)
ax.set_ylabel('Fraction of optimum productivity', fontsize=6)
ax.set_xlabel(r'Temperature ($^\circ$C)', fontsize=6)
ax.axvline(temp_opt, linestyle='--', color='#A9A9A9', linewidth=0.7)
ax.xaxis.set_tick_params(labelsize=5)
ax.yaxis.set_tick_params(labelsize=5)


fig.subplots_adjust(bottom=0.2, left=0.18, top=0.95, right=0.95)

fig.savefig('figures/figure_damage_function.pdf')

#--------------------------------------------------------------------
