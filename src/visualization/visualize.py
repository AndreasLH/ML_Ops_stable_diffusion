from src.data.dataloader import ButterflyDataloader
import matplotlib.pyplot as plt
import click


@click.group()
def cli():
    pass


def grid_func(n,i,j):

    if i > (n // 2) - 1:
        if i == n // 2:
            j += 1
        i -= n // 2

    return i,j

@click.command()
@click.argument('datapath', type=click.Path(exists=True))
@click.argument('n', type=int)
def visualize(datapath, n=8):
    assert n % 2 == 0, "n must be an even number"

    train_dataset = ButterflyDataloader(datapath)

    # plot n images
    j = 0
    fig, axs = plt.subplots(2, int(n/2), figsize=(16, 4))
    for i, image in enumerate(train_dataset[:n]["images"]):
        # use the same transformation back, from [-1,1] to [0,1]
        image = ((image * 0.5) + 0.5)
        # get image coordinates
        i,j = grid_func(n,i,j)

        axs[j,i].imshow(image.permute(1,2,0)) # permute to H x W x Channel, because that is what matplotlib wants
        axs[j,i].set_axis_off()

    plt.show()


if __name__ == '__main__':
    visualize()