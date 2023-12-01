# deep-learning-final-project
Final Project for the Deep Learning course at Aarhus University Autumn 2023

## SSH tutorial lmaooo

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

Now you can run the [pre-processing scripts](#pre-processing-steps).

- [ ] Combine datasets

### Pre-Processing Steps

- [x] Convert to 32-bit float
- [x] Sum channels to mono
- [x] Downsample to 44.1kHz
- [x] Trim or pad to exactly N samples per audio file
- [ ] Detrend
- [x] Declick (fade out last ~100 samples)
- [ ] Normalize
- [x] Save as `<instrument>_<number>` for example `snare_123.wav`.
