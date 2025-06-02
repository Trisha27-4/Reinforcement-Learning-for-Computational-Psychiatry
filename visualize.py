
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_q_values(q_values, title="Q-value Evolution"):
    qa_vals, qb_vals = zip(*q_values)
    plt.plot(qa_vals, label='Q(A)')
    plt.plot(qb_vals, label='Q(B)')
    plt.title(title)
    plt.xlabel("Trial")
    plt.ylabel("Q Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_param_distributions(params, labels, title="Parameter Distribution", group_labels=None):
    params = np.array(params)
    fig, axs = plt.subplots(1, params.shape[1], figsize=(5 * params.shape[1], 4))
    for i in range(params.shape[1]):
        if group_labels is not None:
            sns.boxplot(x=group_labels, y=params[:, i], ax=axs[i])
        else:
            axs[i].hist(params[:, i], bins=15, alpha=0.7)
        axs[i].set_title(f"{labels[i]}")
        axs[i].set_xlabel(labels[i])
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
