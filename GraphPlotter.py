import matplotlib.pyplot as plt

def plotLine(x, y):
    # extract the Seconds and Close columns from the DataFrame
    plt.ylim([100, 100])

    # create a Matplotlib figure and axis
    fig, ax = plt.subplots()


    # plot the Seconds and Close columns
    ax.plot(x, y)

    ax.set_ylim(bottom=0, top=100)


    # set the x-axis label
    ax.set_xlabel('Seconds')

    # set the y-axis label
    ax.set_ylabel('Close')

    # show the plot
    plt.show()
