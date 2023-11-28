# %%
import torch


class Generator(torch.nn.Module):
    """
    The generator is a convolutional neural network that takes as input a random vector and outputs a spectrogram.

    The architecture is is adapted from the one in the GANSynth paper, page 17.
    It does not have as many convolutional layers as the paper. The choice of not having as many layers
    Is to speed up training time.
    """

    def __init__(self, leaky_relu_negative_slope: float) -> None:
        super(Generator, self).__init__()

        assert leaky_relu_negative_slope > 0, f'Expected {leaky_relu_negative_slope = } to be > 0'
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, f'Expected 2 dimensions, got {x.ndim = }'
        batch_size: int = x.shape[0]
        latent_vector_size: int = x.shape[1]
        assert latent_vector_size == 256, f'Expected {latent_vector_size = } to be 256'

        x = self.conv1(x)
        assert x.shape == (
            batch_size,
            256,
            2,
            16,
        ), f'Expected shape ({batch_size}, 256, 2, 16), got {x.shape = }'
        x = self.leaky_relu(x)

        # torch.nn.functional.upsample() or torch.nn.functional.interpolate()?
        assert x.shape == (
            batch_size,
            256,
            4,
            32,
        ), f'Expected shape ({batch_size}, 256, 4, 32), got {x.shape = }'

        x = self.conv2(x)
        assert x.shape == (
            batch_size,
            256,
            4,
            32,
        ), f'Expected shape ({batch_size}, 256, 4, 32), got {x.shape = }'
        x = self.leaky_relu(x)

        assert x.shape == (
            batch_size,
            2,
            128,
            1024,
        ), f'Expected shape ({batch_size}, 2, 128, 1024), got {x.shape = }'
        x = torch.nn.functional.tanh(x)
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

        assert leaky_relu_negative_slope > 0, f'Expected {leaky_relu_negative_slope = } to be > 0'
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)

        self.pool1 = torch.nn.AvgPool2d((2, 2), stride=2)
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(1, 1))

        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1)
        )
        self.pool2 = torch.nn.AvgPool2d((2, 2), stride=2)

        self.conv3 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1)
        )
        self.pool3 = torch.nn.AvgPool2d((2, 2), stride=2)

        self.conv4 = torch.nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1)
        )
        self.pool4 = torch.nn.AvgPool2d((2, 2), stride=2)

        self.conv5 = torch.nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1)
        )
        self.pool5 = torch.nn.AvgPool2d((2, 2), stride=2)

        self.conv6 = torch.nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1)
        )
        self.pool6 = torch.nn.AvgPool2d((2, 2), stride=2)

        # GANSynth does NOT use a fully connected layer at the end, but I do not fully understand how
        # they go from the 2x2x256 tensor to a single scalar value.
        # WaveGAN uses a fully connected layer at the end, so I will do the same.
        self.fc1 = torch.nn.Linear(in_features=256 * 2 * 16, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, channels, height, width)
        assert x.ndim == 4, f'Expected 4 dimensions, got {x.ndim = }'
        batch_size: int = x.shape[0]

        def expect(c: int, h: int, w: int) -> None:
            assert x.shape == (
                batch_size,
                c,
                h,
                w,
            ), f'Expected shape ({batch_size}, {c}, {h}, {w}), got {x.shape = }'

        assert x.shape == (
            batch_size,
            2,
            128,
            1024,
        ), f'Expected shape ({batch_size}, 2, 128, 1024), got {x.shape = }'

        x = self.conv1(x)
        assert x.shape == (
            batch_size,
            32,
            128,
            1024,
        ), f'Expected shape ({batch_size}, 32, 128, 1024), got {x.shape = }'
        x = self.leaky_relu(x)
        x = self.pool1(x)
        assert x.shape == (
            batch_size,
            32,
            64,
            512,
        ), f'Expected shape ({batch_size}, 32, 64, 512), got {x.shape = }'

        x = self.conv2(x)
        assert x.shape == (
            batch_size,
            64,
            64,
            512,
        ), f'Expected shape ({batch_size}, 64, 64, 512), got {x.shape = }'
        x = self.leaky_relu(x)

        x = self.pool2(x)
        assert x.shape == (
            batch_size,
            64,
            32,
            256,
        ), f'Expected shape ({batch_size}, 64, 32, 256), got {x.shape = }'

        x = self.conv3(x)
        assert x.shape == (
            batch_size,
            128,
            32,
            256,
        ), f'Expected shape ({batch_size}, 128, 32, 256), got {x.shape = }'
        x = self.leaky_relu(x)
        x = self.pool3(x)
        assert x.shape == (
            batch_size,
            128,
            16,
            128,
        ), f'Expected shape ({batch_size}, 128, 16, 128), got {x.shape = }'

        x = self.conv4(x)
        assert x.shape == (
            batch_size,
            256,
            16,
            128,
        ), f'Expected shape ({batch_size}, 256, 16, 128), got {x.shape = }'
        x = self.leaky_relu(x)
        x = self.pool4(x)
        assert x.shape == (
            batch_size,
            256,
            8,
            64,
        ), f'Expected shape ({batch_size}, 256, 8, 64), got {x.shape = }'

        x = self.conv5(x)
        assert x.shape == (
            batch_size,
            256,
            8,
            64,
        ), f'Expected shape ({batch_size}, 256, 8, 64), got {x.shape = }'
        x = self.leaky_relu(x)
        x = self.pool5(x)
        assert x.shape == (
            batch_size,
            256,
            4,
            32,
        ), f'Expected shape ({batch_size}, 256, 4, 32), got {x.shape = }'

        x = self.conv6(x)
        assert x.shape == (
            batch_size,
            256,
            4,
            32,
        ), f'Expected shape ({batch_size}, 256, 4, 32), got {x.shape = }'
        x = self.leaky_relu(x)
        x = self.pool6(x)
        assert x.shape == (
            batch_size,
            256,
            2,
            16,
        ), f'Expected shape ({batch_size}, 256, 2, 16), got {x.shape = }'

        x = x.reshape(batch_size, -1)
        # print(f"{x.shape = }")
        x = self.fc1(x)
        return x


def main() -> int:
    from loguru import logger

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = Discriminator(0.2)
    d.to(device)
    # (batch_size, width, height, channels)
    x = torch.randn(1, 128, 1024, 2)

    # PyTorch expects (batch_size, channels, width, height)
    x = x.permute(0, 3, 1, 2)
    y = d(x.to(device))
    logger.info(f'{y.shape = }')

    return 0


if __name__ == '__main__':
    pass
    # import sys
    # main()
    # x = torch.randn(3, 3, names=('N', 'C'))
    # print(f"{x.names = }")
    # sys.exit(main())
