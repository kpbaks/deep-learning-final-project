# deep-learning-final-project
Final Project for the Deep Learning course at Aarhus University Autumn 2023

## Important files

- `dataset.py` - Dataset class for loading audio samples
- `model.py` - Class definition for both Generator and Discriminator
- `model293.py` - Model in the state that it was in run `DEEP-293`, used in (some) of the reports.
- `train.py` - Training loop
- `stft.py` - STFT implementation. Used for converting between time and frequency domain, and doing pre-processing in the STFT domain.
- `app.py` - Script to run a trained model and have it generate a sample

## Running the code

Use the `-h | --help` flag to see all available for each script.

## Dependencies

[pixi](https://prefix.dev/) is used for managing dependencies and virtual environments. To install all dependencies run:
```bash
./install-dependencies.sh
```

---

## SSH tutorial (For Group Members)

Tired of logging into servers? Do this!

Have you generated a public+private key pair for your PC? No? Do this:
```bash
ssh-keygen -t ed25519 -C "<your-email>"
```

Now copy your keys to the cluster
```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub dl25e23@gpucluster.st.lab.au.dk
```

Now you can just ssh like normal and you won't be asked for a password!
```bash
ssh -J dl25e23@gpucluster.st.lab.au.dk dl25e23@node6
```

## Data Preparation

### Sourcing Data

There are some guidelines for selecting samples:
- Not over-produced
- Permissively licensed
- Not terrible
- `.wav` only

### File Format

Sample packs all have different naming schemes, file formats and sample rates. Some manual labour is required to prepare samples for the pre-processing stage.

Samples coming from a sample pack need to be manually separated into folders, the naming should be exactly this for the scripts to work:
- kick
- snare
- chat
- ohat
- clap (unused)
- cym (unused)
- tom (unused)
- other (unused)

They should all be in `.wav` format. If they are in some other format, like `.ogg` or `.flac` they should be converted manually to `.wav`. The preprocessing handles things like bit-depth and sample rate automatically.

Now you can run the pre-processing scripts (`preprocessing/examples.ipynb`).
