"""
File: nsynth.py
Author: Kwon-Young Choi
Email: kwon-young.choi@hotmail.fr
Date: 2018-11-13
Description: Load NSynth dataset using pytorch Dataset.
If you want to modify the output of the dataset, use the transform
and target_transform callbacks as ususal.
"""
import glob
import json
import os
from typing import Callable
from pathlib import Path
import argparse

import numpy as np
import scipy.io.wavfile
import torch
# import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder


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
        assert root.exists() and root.is_dir(), f"{root} is not a valid directory"

        if blacklist_pattern is None:
            blacklist_pattern = []

        if categorical_field_list is None:
            categorical_field_list = ["instrument_family"]

        self.root = root
        self.filenames = glob.glob(os.path.join(root, "audio/*.wav"))

        with open(os.path.join(root, "examples.json"), "r") as f:
            self.json_data = json.load(f)

        for pattern in blacklist_pattern:
            self.filenames, self.json_data = self.blacklist(
                self.filenames, self.json_data, pattern
            )

        self.categorical_field_list = categorical_field_list
        self.labelencoders: list[LabelEncoder] = []

        for i, field in enumerate(self.categorical_field_list):
            self.labelencoders.append(LabelEncoder())
            field_values = [value[field] for value in self.json_data.values()]
            self.labelencoders[i].fit(field_values)

        self.transform = transform
        self.target_transform = target_transform

    def blacklist(self, filenames, json_data, pattern):
        filenames = [filename for filename in filenames if pattern not in filename]
        json_data = {
            key: value for key, value in json_data.items() if pattern not in key
        }
        return filenames, json_data

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> tuple:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (audio sample, *categorical targets, json_data)
        """
        name = self.filenames[index]
        _, sample = scipy.io.wavfile.read(name)
        target = self.json_data[os.path.splitext(os.path.basename(name))[0]]
        categorical_target = [
            le.transform([target[field]])[0]
            for field, le in zip(self.categorical_field_list, self.labelencoders)
        ]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return [sample, *categorical_target, target]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__).removesuffix(".py"))
    parser.add_argument("dataset_dir", type=str, help="dataset directory root. Can be the train, test or valid directory")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    assert dataset_dir.exists() and dataset_dir.is_dir(), f"{dataset_dir} is not a valid directory"

    # audio samples are loaded as an int16 numpy array
    # rescale intensity range as float [-1, 1]
    tofloat = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)
    # use instrument_family and instrument_source as classification targets
    dataset = NSynth(
        dataset_dir,
        # "../nsynth-test",
        transform=tofloat,
        blacklist_pattern=["string"],  # blacklist string instrument
        categorical_field_list=["instrument_family", "instrument_source"],
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for samples, instrument_family_target, instrument_source_target, targets in dataloader:
        print(
            samples.shape,
            instrument_family_target.shape,
            instrument_source_target.shape,
        )
        print(torch.min(samples), torch.max(samples))
