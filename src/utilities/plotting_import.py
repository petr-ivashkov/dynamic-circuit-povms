import pickle
from src.utilities.path import path

# Plotting tools
from qiskit.visualization import plot_histogram, plot_state_city, array_to_latex
import matplotlib.pyplot as plt


# Colors 
blue = '#0000BF'
red = '#B22222'
light_grey = '#e5e5e5e5'
grey = '#A9A9A9'
dark_grey = '#555555'
white = '#FFFFFF'
black = '#000000'

# Golden ratio
figure_size_x = 6.0462
figure_size_y = figure_size_x/1.618

plt.rcParams.update(
    {
        "xtick.direction": "in",
        "ytick.direction": "out",
        "ytick.right": False,
        "xtick.top": False,
        "ytick.left": True,
        "xtick.bottom": False,
        "figure.facecolor": "1",
        "savefig.facecolor": "1",
        "savefig.dpi": 600,
        "figure.dpi": 600,
        "savefig.bbox": "tight",
        "font.size": 7,
        "font.family": "serif",
        "lines.markersize": 6,
        "lines.linewidth": 1,
        'axes.axisbelow' : True
    }
)