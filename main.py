import torch
import numpy as np
import matplotlib.pyplot as plt
from cubic import Cubic_Localminimax
from gda import gda

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)

if __name__ == '__main__':

    train_loss_1, acc_1, robust_acc_1, sample_size_1, time_result_1 = Cubic_Localminimax(gamma=2.0, batch_size=512, num_epochs=200, num_ascent_epochs=20, lr_1=0.1, lr_2=0.01, lr_3=0.1, lr_4=0.002)
    train_loss_2, acc_2, robust_acc_2, sample_size_2, time_result_2 = gda(gamma = 2.0, batch_size = 512, num_epochs = 2000, num_ascent_epochs = 20, lr_1 = 0.1, lr_2 = 0.01)
    

    plt.style.use("ggplot")
    fig = plt.figure(figsize=(14, 7))

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    length = range(len(acc_1))

    ax1.plot(length, train_loss_1, color="blue", lw=2, ls="-", label="Cubic-Localminimax")
    ax1.plot(length, train_loss_2, color="red", lw=2, ls="--", label="GDA")
    ax1.set_xlabel("# of epochs", fontweight="bold", fontsize=20)
    ax1.set_ylabel(r" estimate loss $ \varphi(x) $", fontweight="bold", fontsize=20)
    ax1.legend(fontsize=16)

    ax2.plot(length, robust_acc_1, color="blue", lw=2, ls="-")
    ax2.plot(length, robust_acc_2, color="red", lw=2, ls="--")
    ax2.set_xlabel("# of epochs", fontweight="bold", fontsize=20)
    ax2.set_ylabel("robust test accuracy", fontweight="bold", fontsize=20)
    ax2.legend(fontsize=16, loc="upper left", labels=["cubic gda", "gda"])

    plt.show()
    plt.savefig("save/plot1.png")

