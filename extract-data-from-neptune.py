#!/usr/bin/env -S pixi run python3
# %%
from pathlib import Path

import neptune
from loguru import logger
import matplotlib.pyplot as plt

# %%

# Query data from a run
run: neptune.Run = neptune.Run(
    with_id='DEEP-69',
    project='jenner/deep-learning-final-project',
    api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNzNkMDgxNS1lOTliLTRjNWQtOGE5Mi1lMDI5NzRkMWFjN2MifQ==',
    mode='read-only',
)
logger.info(f'{run.get_url() = }')
logger.info(f'{run.print_structure() = }')

# %%

# print(f"{run['parameters/batch_size'].fetch() = }")


d_accuracy_fake = run['train/accuracy/Discriminator_accuracy_fake'].fetch_values()
d_accuracy_real = run['train/accuracy/Discriminator_accuracy_real'].fetch_values()
d_accuracy_total = run['train/accuracy/Discriminator_accuracy_total'].fetch_values()
g_accuracy = run['train/accuracy/Generator_accuracy'].fetch_values()

d_loss_fake = run['train/error/Discriminator_loss_fake'].fetch_values()
d_loss_real = run['train/error/Discriminator_loss_real'].fetch_values()

d_loss_total = run['train/error/Discriminator_loss_total'].fetch_values()
g_error = run['train/error/Generator_error'].fetch_values()

# %%


# type(accuracy_fake)

fig, ax = plt.subplots()

ax.plot(
    d_accuracy_fake['step'], d_accuracy_fake['value'], label='Discriminator accuracy fake', lw=0.25
)
ax.plot(
    d_accuracy_real['step'], d_accuracy_real['value'], label='Discriminator accuracy real', lw=0.25
)
ax.plot(
    d_accuracy_total['step'],
    d_accuracy_total['value'],
    label='Discriminator accuracy total',
    lw=0.25,
)

# plt.plot(accuracy_fake['step'], accuracy_fake['value'])

fig.show()

# %%

run.close()
