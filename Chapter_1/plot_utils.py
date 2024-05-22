import matplotlib.pyplot as plt


def plot_solution(axs, sigma, eps_total, label):

    axs.plot(eps_total, sigma, '--o', label=label, alpha=0.5)

    axs.legend()

    axs.grid(linestyle='-', linewidth=1)

    axs.set_title('$\sigma$ as a function of $\epsilon$')
    axs.ticklabel_format(style='sci', scilimits=(0, 0))


def plot_evolution_training(list_guess, E_true, list_mcre):
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    axs[0].plot(list_guess, label='$E_{guess}$')
    axs[0].hlines(E_true, xmin=0, xmax=len(list_guess) - 1, color='k', linestyles='--', label='$E_{true}$')
    axs[0].set_title('Evolution E during training')
    axs[0].set_xlabel("Epochs")
    axs[0].ticklabel_format(style='sci', scilimits=(0, 0))
    axs[0].legend(loc='lower right')
    axs[0].grid()

    axs[1].plot(list_mcre)
    axs[1].set_title('Evolution of mCRE during training')
    axs[1].set_xlabel("Epochs")
    axs[1].ticklabel_format(style='sci', scilimits=(0, 0))
    axs[1].grid()
    axs[1].set_yscale('log')
    plt.show()
