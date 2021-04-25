## A list of all the runs from our experiments ##

# Vanilla VAE and beta-VAE experiments
vaain.py --dataset=lines32 --latent_width=2 --depth=16 --latent=16 --train_dir=TRAIN --advweight=0 --beta=1
vaain.py --dataset=lines32 --latent_width=2 --depth=16 --latent=16 --train_dir=TRAIN --advweight=0 --beta=5
vaain.py --dataset=lines32 --latent_width=2 --depth=16 --latent=16 --train_dir=TRAIN --advweight=0 --beta=10
vaain.py --dataset=lines32 --latent_width=2 --depth=16 --latent=16 --train_dir=TRAIN --advweight=0 --beta=15
vaain.py --dataset=lines32 --latent_width=2 --depth=16 --latent=16 --train_dir=TRAIN --advweight=0 --beta=20


# Adversarial VAE
vaain.py --dataset=lines32 --latent_width=2 --depth=16 --latent=16 --train_dir=TRAIN --beta=1
vaain.py --dataset=lines32 --latent_width=2 --depth=16 --latent=16 --train_dir=TRAIN --beta=5
vaain.py --dataset=lines32 --latent_width=2 --depth=16 --latent=16 --train_dir=TRAIN --beta=10
vaain.py --dataset=lines32 --latent_width=2 --depth=16 --latent=16 --train_dir=TRAIN --beta=15
vaain.py --dataset=lines32 --latent_width=2 --depth=16 --latent=16 --train_dir=TRAIN --beta=20


# Adversarial Autoencoder (baseline)
acai.py --dataset=lines32 --latent_width=2 --depth=16 --latent=16 --train_dir=TRAIN
