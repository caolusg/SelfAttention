import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def testPd(x , y):

    instX = pd.Series(x)
    instY = pd.Series(y)

    test = pd.DataFrame([instX, instY], columns=['a', 'b'])

    # print(test)

    plt.figure()
    plt.xticks(range(len(y)), y)
    plt.yticks(())
    test.plot.hexbin(x = 'b', y = 'a', gridsize=25)
    plt.show()