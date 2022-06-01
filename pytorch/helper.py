from pathlib import Path
import matplotlib.pyplot as plt
import pickle


def print_loss():
    fp = Path('pytorch\\loss\\20220520104430.loss')
    with open(fp, 'rb') as handle:
        loss = pickle.load(handle)
    x = range(len(loss))
    plt.plot(x, loss)
    plt.show()
    print(len(x))
    
print_loss()