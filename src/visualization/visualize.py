import click
import matplotlib.pyplot as plt

from src.data.dataloader import ButterflyDataloader


@click.group()
def cli():
    pass


def grid_func(n,i,j):
    """Deprecated, only works for 2 x n."""
    if i > (n // 2) - 1:
        if i == n // 2:
            j += 1
        i -= n // 2

    return i,j

def grid_func2(n,m,i,j):
    if i > (n * m // m) - 1:
        while i > (n * m // m) - 1:
            i -= (n * m // m)
            if i == 0:
                j += 1



    return i,j

@click.command()
@click.argument('datapath', type=click.Path(exists=True))
@click.argument('n', type=int)
@click.argument('m', type=int)
def visualize(datapath, n=6, m=3):
    assert isinstance(n,int) == True, "n must be an integer"
    assert isinstance(m,int) == True, "m must be an integer"
    assert n >= m, "must be n >= m"

    train_dataset = ButterflyDataloader(datapath)

    # plot n images
    j = 0
    fig, axs = plt.subplots(m, n, figsize=(16, 4))
    for i, image in enumerate(train_dataset[:m*n]["images"]):
        # use the same transformation back, from [-1,1] to [0,1]
        image = ((image * 0.5) + 0.5)
        # get image coordinates
        i,j = grid_func2(n,m,i,j)

        axs[j,i].imshow(image.permute(1,2,0)) # permute to H x W x Channel, because that is what matplotlib wants
        axs[j,i].set_axis_off()

    plt.show()


if __name__ == '__main__':
    visualize()