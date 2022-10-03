import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(fish_eatens, mean_fish_eatens,records,fish_deque):
    display.clear_output(wait=True)
    display.display(plt.figure('Fish'))
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Fish Eaten')
    plt.plot(fish_eatens)
    plt.plot(mean_fish_eatens)
    plt.plot(records)
    plt.plot(fish_deque)
    plt.ylim(ymin=0)
    plt.text(len(fish_eatens)-1, fish_eatens[-1], str(fish_eatens[-1]))
    plt.text(len(mean_fish_eatens)-1, mean_fish_eatens[-1], str(mean_fish_eatens[-1]))
    plt.show(block=False)

def plot_time(time_alives, mean_time_alives,time_records):
    
    display.clear_output(wait=True)
    display.display(plt.figure('Time'))
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Time Alive')
    plt.plot(time_alives)
    plt.plot(mean_time_alives)
    plt.plot(time_records)
    plt.ylim(ymin=0)
    plt.text(len(time_alives)-1, time_alives[-1], str(time_alives[-1]))
    plt.text(len(mean_time_alives)-1, mean_time_alives[-1], str(mean_time_alives[-1]))
    plt.show(block=False)
