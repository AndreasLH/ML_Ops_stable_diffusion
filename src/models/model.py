from typing import Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from torchvision import transforms
from torchmetrics.image.inception import InceptionScore


class UNet2DModelPL(pl.LightningModule):
    def __init__(
        self, sample_size: int, learning_rate=1e-3, hpms=None):  # jeg gætter på, det er en tuple
        super().__init__()
        self.lr = learning_rate
        # self.config = config
        self.UNet2DModel = UNet2DModel(
            sample_size=sample_size,  # the target image resolution #DEBUG implementer config
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

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.hpms = hpms

    # a change was made such that forward return torch.tensor instead of Union[UNet2DOutput, Tuple]
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Tuple]:
        output = self.UNet2DModel(sample, timestep)
        if not isinstance(output, Tuple):
            return output.sample

    def training_step(self, batch: int, batch_idx: int) -> torch.Tensor:
        clean_images = batch["images"]
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.num_train_timesteps,
            (bs,),
            device=clean_images.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(
            clean_images, noise, timesteps
        )  # DEBUG her fjernede jeg [0].

        # Predict the noise residual
        noise_pred = self(noisy_images, timesteps, return_dict=False)
        loss = F.mse_loss(noise_pred, noise)

        self.log("train_loss", loss)

        return loss

    def compute_inceptionscore(self, batch : torch.Tensor) -> torch.Tensor:
        _ = torch.manual_seed(self.hpms.seed)
        # normalize False,so batch needs to be in range [0, 255] and dtype uint8
        inception = InceptionScore(normalize=False)
        inception.update(batch)
        inception_mean, _ = inception.compute()

        return inception_mean

    # todo: den her gør bare det samme som training_step. Den skal returne inception score i stedet :)
    def validation_step(self, batch: int, batch_idx: int) -> torch.Tensor:
        images = self.sample(self.hpms.eval_batch_size)

        # transform PIL Image to tensors to compute inception score
        transform = transforms.Compose([transforms.PILToTensor()])
        images_as_tensors = torch.stack([transform(i) for i in images])

        inception_score = self.compute_inceptionscore(images_as_tensors)

        # log inception score
        self.log("inception score", inception_score)

        return inception_score

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr
        )  # DEBUG implementer config

    def sample(self, batch_size=4, seed=0, num_inference_steps=2):
        pipeline = DDPMPipeline(unet=self.UNet2DModel, scheduler=self.noise_scheduler)
        images = pipeline(
            batch_size=batch_size,
            generator=torch.manual_seed(seed),
            num_inference_steps=num_inference_steps,
        ).images
        return images
