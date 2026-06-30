import pylab as plt



def plot_data(X, fX, Y, dirname):
    plt.scatter(X.numpy(), Y.numpy(), alpha = 0.7)
    plt.title("Data points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"figures/{dirname}/data.png")
    plt.close()


def plot_line(X, Y, w, b, dirname):
    plt.scatter(X.numpy(), Y.numpy(), alpha = 0.7)
    plt.plot(X, w*X + b)
    plt.title("Data points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"figures/{dirname}/line.png")
    plt.close()


def plot_loss(losses, dirname):
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Evolution of loss value")
    plt.savefig(f"figures/{dirname}/loss.png")
    plt.close()