"""
Implementation of Utility Functions
"""
import os

DPI = 300
DIR_FIGURE = '../figure'

def save_figure(fig, chap, name):
    fn = os.path.join(DIR_FIGURE, '_'.join([chap, name]))
    fig.savefig(fn, dpi=DPI)