#!/usr/bin/env -S pixi run python3
# %%
import math
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class Metadata:
    id: int
    drum_type: str
    label: str


DATASET_DIR = Path.home() / 'datasets' / 'classic_clean'
assert DATASET_DIR.exists(), f'{DATASET_DIR} does not exist'

TRAIN_DIR = DATASET_DIR / 'train'
TEST_DIR = DATASET_DIR / 'test'
VAL_DIR = DATASET_DIR / 'val'
TRAIN_DIR.mkdir(exist_ok=True)
TEST_DIR.mkdir(exist_ok=True)
VAL_DIR.mkdir(exist_ok=True)

logger.info(f'{DATASET_DIR = }')
logger.info(f'{TRAIN_DIR = }')
logger.info(f'{TEST_DIR = }')
logger.info(f'{VAL_DIR = }')

# Check if {train,test,val} directories are empty
train_dir_is_empty = len(list(TRAIN_DIR.glob('*'))) == 0
test_dir_is_empty = len(list(TEST_DIR.glob('*'))) == 0
val_dir_is_empty = len(list(VAL_DIR.glob('*'))) == 0

if not train_dir_is_empty:
    logger.warning(f'{TRAIN_DIR} is not empty')
    # Remove all files in the train directory
    for file in TRAIN_DIR.glob('*'):
        file.unlink()
    logger.info(f'{TRAIN_DIR} is now empty')
if not test_dir_is_empty:
    logger.warning(f'{TEST_DIR} is not empty')
    for file in TEST_DIR.glob('*'):
        file.unlink()
    logger.info(f'{TEST_DIR} is now empty')
if not val_dir_is_empty:
    logger.warning(f'{VAL_DIR} is not empty')
    for file in VAL_DIR.glob('*'):
        file.unlink()
    logger.info(f'{VAL_DIR} is now empty')


# %%

# Get all the WAV files
wav_files = list(DATASET_DIR.glob('*.wav'))
logger.info(f'{len(wav_files) = }')


def parse_filename(filename: Path) -> Metadata:
    id, drum_type, label = filename.stem.split('_')
    return Metadata(id=int(id), drum_type=drum_type, label=label)


parsed_filenames = [parse_filename(filename) for filename in wav_files]

drum_types_counter = Counter([parsed_filename.drum_type for parsed_filename in parsed_filenames])
labels_counter = Counter([parsed_filename.label for parsed_filename in parsed_filenames])

logger.info(f'{drum_types_counter = }')
logger.info(f'{labels_counter = }')


def test_split_contains_all_drum_types(split: list[Path]) -> bool:
    unique_drum_types_in_split = set()
    for filename in split:
        parsed_filename = parse_filename(filename)
        unique_drum_types_in_split.add(parsed_filename.drum_type)

    contains_all_drum_types = len(unique_drum_types_in_split) == len(drum_types_counter)
    if not contains_all_drum_types:
        logger.error(
            f'Your split is missing these drum types: {set(drum_types_counter.keys()) - unique_drum_types_in_split}'
        )

    return contains_all_drum_types


def test_split_contains_all_labels(split: list[Path]) -> bool:
    unique_labels_in_split = set()
    for filename in split:
        parsed_filename = parse_filename(filename)
        unique_labels_in_split.add(parsed_filename.label)

    contains_all_labels = len(unique_labels_in_split) == len(labels_counter)
    if not contains_all_labels:
        logger.error(
            f'Your split is missing these labels: {set(labels_counter.keys()) - unique_labels_in_split}'
        )

    return contains_all_labels


# Split the dataset into train, test and validation sets

# %%

# 75% train, 12.5% test, 12.5% validation


train_size: int = math.floor(len(wav_files) * 0.75)
test_size: int = math.ceil(len(wav_files) * 0.125)
val_size: int = math.ceil(len(wav_files) * 0.125)

assert train_size + test_size + val_size == len(
    wav_files
), f'{train_size + test_size + val_size = } != {len(wav_files) = }'

logger.info(f'{train_size = } {test_size = } {val_size = }')

random.shuffle(wav_files)  # Shuffle the files once

while True:
    random.shuffle(wav_files)
    train_files = wav_files[:train_size]
    test_files = wav_files[train_size : train_size + test_size]
    val_files = wav_files[train_size + test_size :]
    assert len(train_files) + len(test_files) + len(val_files) == len(wav_files)
    if all(
        [
            test_split_contains_all_drum_types(train_files),
            test_split_contains_all_drum_types(test_files),
            test_split_contains_all_drum_types(val_files),
        ]
        + [
            test_split_contains_all_labels(train_files),
            test_split_contains_all_labels(test_files),
            test_split_contains_all_labels(val_files),
        ]
    ):
        break
    logger.warning('Reshuffling the files because one of the splits is missing a drum type')

logger.info('All splits contain all drum types')

# %%

# Copy the files to the train, test and validation directories
logger.info('copying files to train, test and validation directories')
for file in train_files:
    shutil.copy(file, TRAIN_DIR)

for file in test_files:
    shutil.copy(file, TEST_DIR)

for file in val_files:
    shutil.copy(file, VAL_DIR)

logger.info('all done!')


# %%
