import os
from typing import Callable, List, Union

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, notebook_launcher
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.hub_utils import init_git_repo, push_to_hub
from diffusers.optimization import get_cosine_schedule_with_warmup
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms
from tqdm.auto import tqdm

from src.data.dataset import ButterflyDataset

wandb.init(name="Yucheng", project="mlopsproject21")


# Setup config
@hydra.main(config_path="../../conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: dict) -> None:
    """
    Runs training for the denoising diffusion probabilistic model.
    :param cfg: config file containing the hyperparameters
    :return: None
    """
    config = cfg.experiment["hyperparameters"]

    torch.manual_seed(config.seed)  # Set seed

    # load dataset
    datapath = config.datapath
    train_dataset = ButterflyDataset(datapath)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True
    )

    # Define diffusion model
    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(
            128,
            128,
            256,
            256,
            512,
            512,
        ),  # the number of output channes for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    model = model.to("cpu")

    # Define denoising scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=2)

    #  Training #

    # optimiser
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    # helper functions - take from somewhere else?

    def compute_inceptionscore(batch: torch.Tensor) -> torch.Tensor:
        """
        Computes the inception score for a batch of images
        :param batch: Batch of images
        :return: inception score stored as a torch Tensor
        """
        _ = torch.manual_seed(config.seed)
        # normalize False,so batch needs to be in range [0, 255] and dtype uint8
        inception = InceptionScore(normalize=False)
        inception.update(batch)
        inception_mean, _ = inception.compute()

        return inception_mean

    def make_grid(images: List, rows: int, cols: int) -> Image.Image:
        """
        Creates grid of images for predictions of the model.
        :param images:
        :param rows: Number of rows in grid
        :param cols: Number of columns in grid
        :return: PIL Image object
        """

        w, h = images[0].size
        grid = Image.new("RGB", size=(cols * w, rows * h))
        for i, image in enumerate(images):
            grid.paste(image, box=(i % cols * w, i // cols * h))
        return grid

    def evaluate(
        config: dict, epoch: int, pipeline: Callable, global_step: int
    ) -> None:
        """
        Generate images from random noise
        :param config: dict from config file containing hyperparameters
        :param epoch: Number of epoch
        :param pipeline: pipeline object
        :param global_step: global step parameter used for logging
        :return: None
        """
        # Sample some images from random noise (this is the backward diffusion process).
        # The default pipeline output type is `List[PIL.Image]`
        images = pipeline(
            batch_size=config.eval_batch_size,
            generator=torch.manual_seed(config.seed),
        ).images

        # transform PIL Image to tensors to compute inception score
        transform = transforms.Compose([transforms.PILToTensor()])
        images_as_tensors = torch.stack([transform(i) for i in images])

        inception_score = compute_inceptionscore(images_as_tensors)

        # log inception score
        wandb.log(
            data={"inception score": inception_score.detach().item()}, step=global_step
        )

        # Make a grid out of the images
        image_grid = make_grid(images, rows=4, cols=4)

        # Save the images
        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")

    # train loop
    def train_loop(
        config,
        model: nn.Module,
        noise_scheduler: Union[Callable, nn.Module],
        optimizer: Union[Callable, nn.Module],
        train_dataloader: DataLoader,
        lr_scheduler: Union[Callable, nn.Module],
    ) -> None:
        """
        Main training loop
        :param config: dict from config file containing hyperparameters
        :param model: the nn.Module U-Net model
        :param noise_scheduler: DDPM scheduler
        :param optimizer: AdamW optimiser
        :param train_dataloader: torch dataloader class
        :param lr_scheduler: learning rate scheduler
        :return: None
        """
        # Initialize accelerator and tensorboard logging
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            logging_dir=os.path.join(config.output_dir, "logs"),
        )
        if accelerator.is_main_process:
            if config.push_to_hub:
                repo = init_git_repo(config, at_init=True)
            accelerator.init_trackers("train_example")

        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

        global_step = 0

        # Now you train the model
        for epoch in range(config.num_epochs):
            progress_bar = tqdm(
                total=len(train_dataloader),
                disable=not accelerator.is_local_main_process,
            )
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                clean_images = batch["images"]
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape).to("cpu")
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (bs,), device="cpu"
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = noise_scheduler.add_noise(
                    clean_images.to("cpu"), noise, timesteps
                )

                with accelerator.accumulate(model):
                    # Predict the noise residual
                    model = model.to("cpu")
                    noise_pred = model(
                        noisy_images.to("cpu"), timesteps, return_dict=False
                    )[0]
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                wandb.log(
                    data={
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
                )

                if (
                    epoch + 1
                ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                    evaluate(config, epoch, pipeline, global_step)

                if (
                    epoch + 1
                ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    if config.push_to_hub:
                        push_to_hub(
                            config,
                            pipeline,
                            repo,
                            commit_message=f"Epoch {epoch}",
                            blocking=True,
                        )
                    else:
                        pipeline.save_pretrained(config.output_dir)

    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

    print(config.num_epochs)

    # train
    notebook_launcher(train_loop, args, num_processes=0)


if __name__ == "__main__":
    main()
