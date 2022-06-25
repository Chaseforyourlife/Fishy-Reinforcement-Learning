import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(fish_eatens, mean_fish_eatens,records):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Fish Eaten')
    plt.plot(fish_eatens)
    plt.plot(mean_fish_eatens)
    plt.plot(records)
    plt.ylim(ymin=0)
    plt.text(len(fish_eatens)-1, fish_eatens[-1], str(fish_eatens[-1]))
    plt.text(len(mean_fish_eatens)-1, mean_fish_eatens[-1], str(mean_fish_eatens[-1]))
    plt.show(block=False)
