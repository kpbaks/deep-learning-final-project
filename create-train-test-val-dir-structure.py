#!/usr/bin/env -S pixi run python3
# %%
import math
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
# import sys

from loguru import logger


@dataclass
class Metadata:
    id: int
    drum_type: str


DATASET_DIR = Path.home() / 'datasets' / 'drums'
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
    tags = filename.stem.split('_')
    id = tags[0]
    drum_type = tags[1]
    return Metadata(id=int(id), drum_type=drum_type)


parsed_filenames = [parse_filename(filename) for filename in wav_files]

drum_types_counter = Counter([parsed_filename.drum_type for parsed_filename in parsed_filenames])

logger.info(f'{drum_types_counter = }')


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


# Split the dataset into train, test and validation sets

# %%

# 75% train, 12.5% test, 12.5% validation


total_size: int = len(wav_files)
train_size: int = math.floor(total_size * 0.75)
test_size: int = math.floor(total_size * 0.125)
val_size: int = total_size - train_size - test_size

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
