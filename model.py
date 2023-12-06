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
        self, latent_size: int, pitch_conditioning_size: int, leaky_relu_negative_slope: float
    ) -> None:
        super(Generator, self).__init__()

        assert latent_size > 0, f'Expected {latent_size = } to be > 0'
        assert pitch_conditioning_size > 0, f'Expected {pitch_conditioning_size = } to be > 0'
        self.latent_size = latent_size
        self.pitch_conditioning_size = pitch_conditioning_size
        assert leaky_relu_negative_slope > 0, f'Expected {leaky_relu_negative_slope = } to be > 0'

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.pixel_norm = PixelNorm()

        # (batch_size, latent_size + pitch_conditioning_size, 1, 1)
        # to (batch_size, 256, 2, 16)

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=latent_size + pitch_conditioning_size,
            out_channels=256,
            kernel_size=(2, 16),
            bias=False,
        )

        # self.conv2 = torch.nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), padding="same", bias=False)
        self.conv2 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same', bias=False)

        self.conv3 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same', bias=False)

        self.conv4 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same', bias=False)

        self.conv5 = torch.nn.Conv2d(256, 128, kernel_size=(3, 3), padding='same', bias=False)

        self.conv6 = torch.nn.Conv2d(128, 64, kernel_size=(3, 3), padding='same', bias=False)

        self.conv7 = torch.nn.Conv2d(64, 32, kernel_size=(3, 3), padding='same', bias=False)

        self.conv8 = torch.nn.Conv2d(32, 2, kernel_size=(1, 1), padding='same', bias=False)

    def expected_input_shape(self) -> torch.Size:
        return torch.Size([1, self.latent_size + self.pitch_conditioning_size, 1, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f'Expected 2 dimensions, got {x.ndim = }'
        batch_size: int = x.shape[0]
        latent_vector_size: int = x.shape[1]
        assert (
            latent_vector_size == self.latent_size + self.pitch_conditioning_size
        ), f'Expected {latent_vector_size = } to be {self.latent_size + self.pitch_conditioning_size = }'
        num_except_calls: int = 0

        def expect(c: int, h: int, w: int) -> None:
            """Helper function to check the shape of the tensor at different points in the network."""
            nonlocal num_except_calls
            num_except_calls += 1
            if x.shape != (batch_size, c, h, w):
                raise ValueError(
                    f'expect nr. {num_except_calls}: Expected shape ({batch_size}, {c}, {h}, {w}), got {x.shape = }'
                )

        def a(x):
            return self.pixel_norm(self.leaky_relu(x))

        x = self.conv1(x)
        expect(256, 2, 16)
        x = a(x)

        # Upsample to (batch_size, 256, 4, 32)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 2), mode='nearest')
        expect(256, 4, 32)

        x = self.conv2(x)
        expect(256, 4, 32)
        x = a(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 2), mode='nearest')
        expect(256, 8, 64)

        x = self.conv3(x)
        expect(256, 8, 64)
        x = a(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 2), mode='nearest')
        expect(256, 16, 128)

        x = self.conv4(x)
        expect(256, 16, 128)
        x = a(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 2), mode='nearest')
        expect(256, 32, 256)

        x = self.conv5(x)
        expect(128, 32, 256)
        x = a(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 2), mode='nearest')
        expect(128, 64, 512)

        x = self.conv6(x)
        expect(64, 64, 512)
        x = a(x)

        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1), mode='nearest')
        expect(64, 128, 512)
        x = self.conv7(x)
        expect(32, 128, 512)
        x = a(x)

        x = self.conv8(x)
        expect(2, 128, 512)
        # TODO: maybe batch normalize the output?
        x = torch.nn.functional.tanh(x)

        # # NOTE: in the GANSynth paper, they say they use "2x2 box upsampling"
        # # torch.nn.functional.upsample() or torch.nn.functional.interpolate()?

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

        bias: bool = False

        assert leaky_relu_negative_slope > 0, f'Expected {leaky_relu_negative_slope = } to be > 0'
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)

        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(1, 1), bias=bias)
        self.pool1 = torch.nn.AvgPool2d((2, 2), stride=2)

        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), bias=bias, padding=(1, 1)
        )
        self.pool2 = torch.nn.AvgPool2d((2, 2), stride=2)

        self.conv3 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), bias=bias, padding=(1, 1)
        )
        self.pool3 = torch.nn.AvgPool2d((2, 2), stride=2)

        self.conv4 = torch.nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=(3, 3), bias=bias, padding=(1, 1)
        )
        self.pool4 = torch.nn.AvgPool2d((2, 2), stride=2)

        self.conv5 = torch.nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), bias=bias, padding=(1, 1)
        )
        self.pool5 = torch.nn.AvgPool2d((2, 2), stride=2)

        self.conv6 = torch.nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), bias=bias, padding=(1, 1)
        )
        self.pool6 = torch.nn.AvgPool2d((2, 2), stride=2)

        # GANSynth does NOT use a fully connected layer at the end, but I do not fully understand how
        # they go from the 2x2x256 tensor to a single scalar value.
        # WaveGAN uses a fully connected layer at the end, so I will do the same.
        # self.fc1 = torch.nn.Linear(in_features=256 * 2 * 16, out_features=1)

        self.conv7 = torch.nn.Conv2d(
            in_channels=256, out_channels=12, kernel_size=(2, 8), bias=bias
        )
        self.sigmoid = torch.nn.Sigmoid()

        self.conv8 = torch.nn.Conv2d(in_channels=12, out_channels=1, kernel_size=(1, 1), bias=bias)

    def expected_input_shape(self) -> torch.Size:
        return torch.Size([2, 128, 512])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, channels, height, width)
        assert x.ndim == 4, f'Expected 4 dimensions, got {x.ndim = }'
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

        x = self.conv1(x)
        expect(32, 128, 512)
        x = self.leaky_relu(x)
        x = self.pool1(x)
        expect(32, 64, 256)

        x = self.conv2(x)
        expect(64, 64, 256)
        x = self.leaky_relu(x)
        x = self.pool2(x)
        expect(64, 32, 128)

        x = self.conv3(x)
        expect(128, 32, 128)
        x = self.leaky_relu(x)
        x = self.pool3(x)
        expect(128, 16, 64)

        x = self.conv4(x)
        expect(256, 16, 64)
        x = self.leaky_relu(x)
        x = self.pool4(x)
        expect(256, 8, 32)

        x = self.conv5(x)
        expect(256, 8, 32)
        x = self.leaky_relu(x)
        x = self.pool5(x)
        expect(256, 4, 16)

        x = self.conv6(x)
        expect(256, 4, 16)
        x = self.leaky_relu(x)
        x = self.pool6(x)
        expect(256, 2, 8)

        # Global average pooling

        x = self.conv7(x)
        x = torch.nn.functional.softmax(x, dim=1)
        # x = self.sigmoid(x)

        x = self.conv8(x)
        # x = x.reshape(batch_size, -1)
        # print(f"{x.shape = }")
        # x = self.fc1(x)

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

    # Discriminate
    # (batch_size, width, height, channels)
    # x = torch.randn(1, 128, 512, 2)

    # PyTorch expects (batch_size, channels, width, height)
    # x = x.permute(0, 3, 1, 2)
    discrimination = d(y.to(device))
    logger.info(f'{discrimination.shape = }')

    t_diff = time.time() - t_start

    logger.info(f'{t_diff = }')

    return 0


if __name__ == '__main__':
    # pass
    # import sys
    main()
    # x = torch.randn(3, 3, names=('N', 'C'))
    # print(f"{x.names = }")

    # sys.exit(main())
