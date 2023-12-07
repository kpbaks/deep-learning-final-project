# %%
import torch
import time


class PixelNorm(torch.nn.Module):
    def __init__(self, epsilon: float = 1e-8) -> None:
        """
        PixelNorm is a normalization layer.
        The function is defined as (page 16 in GANSynth paper):
        x = x_{nhwc} / (\sqrt{1/C \sum_{c}x^2_{nhwc}} + \epsilon)
        Parameters:
        epsilon (float): A small value added to the denominator for numerical stability.
                         Defaults to 1e-8.
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PixelNorm layer.
        Parameters:
        x (Tensor): The input tensor with shape (n, c, h, w), where n is the batch size,
                    c is the number of channels, h is the height, and w is the width.

        Returns:
        Tensor: The normalized tensor of the same shape as the input.
        """
        assert x.ndim == 4, f'Expected 4 dimensions, got {x.ndim = }'
        normalized = x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)
        assert (
            normalized.shape == x.shape
        ), f'Expected shape {x.shape = }, got {normalized.shape = }'
        return normalized


class Generator(torch.nn.Module):
    """
    The generator is a convolutional neural network that takes as input a random vector and outputs a spectrogram.

    The architecture is is adapted from the one in the GANSynth paper, page 17.
    It does not have as many convolutional layers as the paper. The choice of not having as many layers
    Is to speed up training time.
    """

    def __init__(
        self, latent_size: int, label_conditioning_size: int, leaky_relu_negative_slope: float
    ) -> None:
        super(Generator, self).__init__()

        if latent_size <= 0:
            raise ValueError(f'Expected {latent_size = } to be > 0')
        self.latent_size = latent_size
        if label_conditioning_size <= 0:
            raise ValueError(f'Expected {label_conditioning_size = } to be > 0')
        self.label_conditioning_size = label_conditioning_size
        if leaky_relu_negative_slope <= 0:
            raise ValueError(f'Expected {leaky_relu_negative_slope = } to be > 0')

        # TODO: maybe use embedding layer for pitch conditioning?
        # self.pixel_norm = PixelNorm()

        self.layers = torch.nn.Sequential(
            # (batch_size, latent_size + label_conditioning_size, 1, 1)
            torch.nn.ConvTranspose2d(
                in_channels=latent_size + label_conditioning_size,
                out_channels=256,
                kernel_size=(4, 16),
                bias=False,
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope, inplace=True),
            # (batch_size, 256, 4, 16)
            torch.nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope, inplace=True),
            # (batch_size, 256, 8, 32)
            torch.nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope, inplace=True),
            # # (batch_size, 128, 16, 64)
            torch.nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            ),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope, inplace=True),
            # # (batch_size, 64, 32, 128)
            torch.nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope, inplace=True),
            # # (batch_size, 32, 64, 256)
            torch.nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            ),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope, inplace=True),
            # (batch_size, 16, 128, 512)
            torch.nn.ConvTranspose2d(
                in_channels=32,
                out_channels=2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            torch.nn.Tanh(),
            # (batch_size, 2, 128, 512)
        )

    def expected_input_shape(self) -> torch.Size:
        return torch.Size([self.latent_size + self.label_conditioning_size, 1, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f'Expected 4 dimensions, got {x.ndim = }')
        batch_size: int = x.shape[0]
        latent_vector_size: int = x.shape[1]
        assert (
            latent_vector_size == self.latent_size + self.label_conditioning_size
        ), f'Expected {latent_vector_size = } to be {self.latent_size + self.label_conditioning_size = }'
        num_except_calls: int = 0

        def expect(c: int, h: int, w: int) -> None:
            """Helper function to check the shape of the tensor at different points in the network."""
            nonlocal num_except_calls
            num_except_calls += 1
            if x.shape != (batch_size, c, h, w):
                raise ValueError(
                    f'expect nr. {num_except_calls}: Expected shape ({batch_size}, {c}, {h}, {w}), got {x.shape = }'
                )

        expect(latent_vector_size, 1, 1)
        x = self.layers(x)
        expect(2, 128, 512)
        return x


class Discriminator(torch.nn.Module):
    """
    The discriminator is a convolutional neural network that takes as input a spectrogram and outputs a single scalar value.

    The architecture is is adapted from the one in the GANSynth paper, page 17.
    It does not have as many convolutional layers as the paper. The choice of not having as many layers
    Is to speed up training time.
    """

    def __init__(self, leaky_relu_negative_slope: float) -> None:
        super(Discriminator, self).__init__()
        # TODO: maybe have some dropout layers?
        # TODO: maybe have some normalization layers?

        assert leaky_relu_negative_slope > 0, f'Expected {leaky_relu_negative_slope = } to be > 0'

        # # GANSynth does NOT use a fully connected layer at the end, but I do not fully understand how
        # # they go from the 2x2x256 tensor to a single scalar value.
        # # WaveGAN uses a fully connected layer at the end, so I will do the same.

        self.layers = torch.nn.Sequential(
            # (batch_size, 2, 128, 512)
            torch.nn.Conv2d(
                in_channels=2,
                out_channels=32,
                kernel_size=(1, 1),
                bias=False,
            ),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope, inplace=True),
            torch.nn.AvgPool2d((2, 2), stride=2),
            # (batch_size, 32, 64, 256)
            torch.nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(3, 3), bias=False, padding=(1, 1)
            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope, inplace=True),
            torch.nn.AvgPool2d((2, 2), stride=2),
            # (batch_size, 64, 32, 128)
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                bias=False,
                padding=(1, 1),
            ),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope, inplace=True),
            torch.nn.AvgPool2d((2, 2), stride=2),
            # (batch_size, 128, 16, 64)
            torch.nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=(3, 3), bias=False, padding=(1, 1)
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope, inplace=True),
            torch.nn.AvgPool2d((2, 2), stride=2),
            # (batch_size, 256, 8, 32)
            torch.nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=(3, 3), bias=False, padding=(1, 1)
            ),
            torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope, inplace=True),
            torch.nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=(3, 3), bias=False, padding=(1, 1)
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope, inplace=True),
            torch.nn.AvgPool2d((2, 2), stride=2),
            # (batch_size, 256, 4, 16)
            torch.nn.Conv2d(
                in_channels=256, out_channels=12, kernel_size=(3, 3), bias=False, padding=(1, 1)
            ),
            torch.nn.BatchNorm2d(12),
            torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope, inplace=True),
            torch.nn.AvgPool2d((2, 2), stride=2),
            # (batch_size, 12, 2, 4)
            torch.nn.Conv2d(in_channels=12, out_channels=1, kernel_size=(2, 8), bias=False),
            # (batch_size, 1, 1, 1)
            torch.nn.Sigmoid(),
        )

    def expected_input_shape(self) -> torch.Size:
        return torch.Size([2, 128, 512])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, channels, height, width)
        if x.ndim != 4:
            raise ValueError(f'Expected 4 dimensions, got {x.ndim = }')
        batch_size: int = x.shape[0]

        num_except_calls: int = 0

        def expect(c: int, h: int, w: int) -> None:
            """Helper function to check the shape of the tensor at different points in the network."""
            nonlocal num_except_calls
            num_except_calls += 1
            if x.shape != (batch_size, c, h, w):
                raise ValueError(
                    f'expect nr. {num_except_calls}: Expected shape ({batch_size}, {c}, {h}, {w}), got {x.shape = }'
                )

        expect(2, 128, 512)
        x = self.layers(x)
        expect(1, 1, 1)
        return x


def main() -> int:
    from loguru import logger

    seed: int = 1234
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = Discriminator(0.2)
    d.to(device)

    latent_size: int = 256
    pitch_conditioning_size: int = 3 * 4
    g = Generator(latent_size, pitch_conditioning_size, 0.2)
    g.to(device)

    t_start = time.time()

    # Generate
    z = torch.randn(1, 1, 1, latent_size + pitch_conditioning_size)
    z = z.permute(0, 3, 1, 2)
    z = z.to(device)
    logger.info(f'{z.shape = }')

    y = g(z)
    logger.info(f'{y.shape = }')
    # PyTorch expects (batch_size, channels, width, height)
    discrimination = d(y.to(device))
    logger.info(f'{discrimination.shape = }')

    t_diff = time.time() - t_start

    logger.info(f'{t_diff = }')

    return 0


if __name__ == '__main__':
    main()


# %%
