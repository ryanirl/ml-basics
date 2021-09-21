# CREDITS: https://towardsdatascience.com/basics-of-gifs-with-pythons-matplotlib-54dd544b6f30
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

filenames = ['{}.png'.format(i) for i in range(1, 61)]
print(filenames)

with imageio.get_writer('mygif0.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
