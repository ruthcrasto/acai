## A list of all the runs from our experiments ##

# TODO: add different datasets and latent dimensions

# Vanilla VAE and beta-VAE experiments
vaain.py --dataset=lines32 --latent_width=2 --depth=16 --latent=16 --train_dir=TRAIN --advweight=0 --beta=1
vaain.py --dataset=lines32 --latent_width=2 --depth=16 --latent=16 --train_dir=TRAIN --advweight=0 --beta=10
vaain.py --dataset=lines32 --latent_width=2 --depth=16 --latent=16 --train_dir=TRAIN --advweight=0 --beta=100
vaain.py --dataset=lines32 --latent_width=2 --depth=16 --latent=16 --train_dir=TRAIN --advweight=0 --beta=250

# Adversarial VAE
vaain.py --dataset=lines32 --latent_width=2 --depth=16 --latent=16 --train_dir=TRAIN --beta=1
vaain.py --dataset=lines32 --latent_width=2 --depth=16 --latent=16 --train_dir=TRAIN --beta=100
vaain.py --dataset=lines32 --latent_width=2 --depth=16 --latent=16 --train_dir=TRAIN --advweight=0.8 --beta=100


