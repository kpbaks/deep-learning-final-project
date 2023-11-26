import argparse
import json
import os
import time
from glob import glob
from pathlib import Path
from typing import Callable
from dataclasses import dataclass, asdict

import numpy as np
import scipy.io.wavfile
import torch

# import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from torch import Tensor

try:
    from rich import pretty, print

    pretty.install()
except ImportError or ModuleNotFoundError:
    pass


class NSynth(torch.utils.data.Dataset):

    """Pytorch dataset for NSynth dataset
    args:
        root: root dir containing examples.json and audio directory with
            wav files.
        transform (callable, optional): A function/transform that takes in
                a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        blacklist_pattern: list of string used to blacklist dataset element.
            If one of the string is present in the audio filename, this sample
            together with its metadata is removed from the dataset.
        categorical_field_list: list of string. Each string is a key like
            instrument_family that will be used as a classification target.
            Each field value will be encoding as an integer using sklearn
            LabelEncoder.
    """

    def __init__(
        self,
        root: Path,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        blacklist_pattern: list[str] | None = None,
        categorical_field_list: list[str] | None = None,
    ):
        assert root.exists() and root.is_dir(), f"{root} is not a directory on disk"
        assert (root / "examples.json").exists(), f"{root}/examples.json does not exist"

        if blacklist_pattern is None:
            blacklist_pattern = []

        if categorical_field_list is None:
            categorical_field_list = ["instrument_family"]

        self.root = root
        # TODO: maybe make absolute path
        self.waw_files: list[Path] = [Path(wav) for wav in glob(f"{root}/audio/*.wav")]
        # print(f"{self.waw_files = }")
        # self.filenames = glob.glob(os.path.join(root, "audio/*.wav"))

        with open(root / "examples.json", "r") as f:
            self.json_data = json.load(f)

        # remove blacklisted samples
        i: int = 0
        indices_of_files_to_remove: list[int] = []
        while i < len(self.waw_files):
            for pattern in blacklist_pattern:
                if pattern in self.waw_files[i].name:
                    indices_of_files_to_remove.append(i)
                    break
            i += 1

        self.json_data = {
            k: v
            for k, v in self.json_data.items()
            if any(pattern not in k for pattern in blacklist_pattern)
        }
        # remove blacklisted samples
        # NOTE: We need to remove the files in reverse order to avoid index out of range error
        # due to how the del keyword works with lists
        for index in indices_of_files_to_remove[::-1]:
            del self.waw_files[index]

        # for pattern in blacklist_pattern:
        #     self.waw_files

        # self.waw_files, self.json_data = self.blacklist(
        #     self.waw_files, self.json_data, pattern
        # )

        self.categorical_field_list = categorical_field_list
        self.labelencoders: list[LabelEncoder] = []

        for i, field in enumerate(self.categorical_field_list):
            field_values = [value[field] for value in self.json_data.values()]
            le = LabelEncoder()
            le.fit(field_values)
            self.labelencoders.append(le)
            # self.labelencoders.append(LabelEncoder())
            # self.labelencoders[i].fit(field_values)

        self.transform = transform
        self.target_transform = target_transform

    # def blacklist(self, filenames, json_data, pattern):
    #     filenames = [filename for filename in filenames if pattern not in filename]
    #     json_data = {
    #         key: value for key, value in json_data.items() if pattern not in key
    #     }
    #     return filenames, json_data

    def __len__(self) -> int:
        return len(self.waw_files)

    def __getitem__(
        self, index: int
    ) -> tuple[Tensor, int, int, dict[str, int | str | list[int]]]:
        assert 0 <= index < len(self.waw_files)
        wav_file: Path = self.waw_files[index]
        _, sample = scipy.io.wavfile.read(wav_file)
        sample = torch.from_numpy(sample)
        # print(f"{type(sample) = }")
        # bar = os.path.splitext(wav_file)
        # print(f"{bar = }")
        # foo = os.path.splitext(os.path.basename(wav_file))[0]
        # print(f"{foo = }")
        key = wav_file.name.removesuffix(".wav")
        # print(f"{key = }")
        target = self.json_data[key]
        categorical_target = [
            le.transform([target[field]])[0]
            for field, le in zip(self.categorical_field_list, self.labelencoders)
        ]
        instrument_family_target: int = categorical_target[0]
        instrument_source_target: int = categorical_target[1]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # print(f"{type(target) = }")
        # print(f"{target = }")
        # print(f"{categorical_target = }")
        return (sample, instrument_family_target, instrument_source_target, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__).removesuffix(".py")
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="dataset directory root. Can be the train, test or valid directory",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    assert (
        dataset_dir.exists() and dataset_dir.is_dir()
    ), f"{dataset_dir} is not a valid directory"

    # audio samples are loaded as an int16 numpy array
    # rescale intensity range as float [-1, 1]
    tofloat = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)
    # use instrument_family and instrument_source as classification targets
    dataset = NSynth(
        dataset_dir,
        transform=tofloat,
        blacklist_pattern=["string"],  # blacklist string instrument
        categorical_field_list=["instrument_family", "instrument_source"],
    )

    # NOTE: The collate_fn was added to fix the workaround discussed in the issue below
    # https://github.com/pytorch/pytorch/issues/42654#issuecomment-1000630232
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )

    for (
        samples,
        instrument_family_target,
        instrument_source_target,
        targets,
    ) in dataloader:
        print(f"{samples[0].shape = }")
        print(f"{len(instrument_family_target) = }")
        print(f"{len(instrument_source_target) = }")
        print(f"{targets = }")
