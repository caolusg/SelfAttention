import pandas as pd
import matplotlib.pyplot as plt
def vision(x, y):#x为行元素，y为列元素

    # dataDict = {}
    # for i in range(len(x)):
    #     dataDict[y[i]] = x[i]
    #
    # pic = pd.DataFrame(dataDict)
    #
    # pd.options.display.mpl_style = 'default'
    #
    # pic_plot = pic.plot(kind = "bar", x = pic["detail"],
    #                     title = "TEST", legend = False)
    #
    # fig = pic_plot.get_figure()
    #
    # fig.show()
    plt.bar(y, x)
    plt.xticks(rotation = 45)
    plt.grid()
    plt.show()