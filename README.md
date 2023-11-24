# deep-learning-final-project
Final Project for the Deep Learning course at Aarhus University Autumn 2023

64000

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
