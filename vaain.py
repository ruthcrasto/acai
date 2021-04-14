# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
""" Variational Autoencoder Adversarial Network (VAAIN)
An adversarial beta-VAE for latent-space interpolation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import app
from absl import flags

import tensorflow as tf
from lib import data, layers, train, utils, classifiers, eval

FLAGS = flags.FLAGS


class VAAIN(train.AE):

    def model(self, latent, depth, scales, beta, advweight, advdepth, reg):
        """
        Args:
            latent: number of channels output by the encoder.
            depth: depth (number of channels before applying the first convolution)
                for the encoder
            scales: input width/height to latent width/height ratio, on log base 2 scale
                (how many times the encoder should downsample)
            beta: scale hyperparam >= 1 for the KL term in the ELBO
                (value of 1 equivalent to vanilla VAE)
            advweight: how much the VAE should care about fooling the discriminator
                (value of 0 equivalent to training a VAE alone)
            advdepth: depth for the discriminator
            reg: gamma in the paper
        """
        x = tf.placeholder(tf.float32,
                           [None, self.height, self.width, self.colors], 'x')
        l = tf.placeholder(tf.float32, [None, self.nclass], 'label')
        h = tf.placeholder(
            tf.float32,
            [None, self.height >> scales, self.width >> scales, latent], 'h')

        def encoder(x):
            """ Outputs latent codes (not mean vectors) """
            return layers.encoder(x, scales, depth, latent, 'ae_enc')

        def decoder(h):
            """ Outputs Bernoulli logits """
            return layers.decoder(h, scales, depth, self.colors, 'ae_dec')

        def disc(x):
            """ Outputs predicted mixing coefficient alpha """
            return tf.reduce_mean(
                layers.encoder(x, scales, advdepth, latent, 'disc'),
                axis=[1, 2, 3])

        # ENCODE
        encode = encoder(x)
        # get mean and var from the latent code
        with tf.variable_scope('ae_latent'):
            encode_shape = tf.shape(encode)
            encode_flat = tf.layers.flatten(encode)
            latent_dim = encode_flat.get_shape()[-1]
            q_mu = tf.layers.dense(encode_flat, latent_dim)
            log_q_sigma_sq = tf.layers.dense(encode_flat, latent_dim)
        # sample
        q_sigma = tf.sqrt(tf.exp(log_q_sigma_sq))
        q_z = tf.distributions.Normal(loc=q_mu, scale=q_sigma)
        q_z_sample = q_z.sample()
        q_z_sample_reshaped = tf.reshape(q_z_sample, encode_shape)

        # DECODE
        p_x_given_z_logits = decoder(q_z_sample_reshaped)
        vae = 2*tf.nn.sigmoid(p_x_given_z_logits) - 1  # [0, 1] -> [-1, 1]
        decode = 2 * tf.nn.sigmoid(decoder(h)) - 1

        # COMPUTE VAE LOSS
        p_x_given_z = tf.distributions.Bernoulli(logits=p_x_given_z_logits)
        loss_kl = 0.5 * tf.reduce_sum(
            -log_q_sigma_sq - 1 + tf.exp(log_q_sigma_sq) + q_mu ** 2)
        loss_kl = loss_kl / tf.to_float(tf.shape(x)[0])
        x_bernoulli = 0.5 * (x + 1)  # [-1, 1] -> [0, 1]
        loss_ll = tf.reduce_sum(p_x_given_z.log_prob(x_bernoulli))
        loss_ll = loss_ll / tf.to_float(tf.shape(x)[0])
        elbo = loss_ll - beta * loss_kl
        loss_vae = -elbo
        utils.HookReport.log_tensor(loss_vae, 'neg elbo')

        # COMPUTE DISCRIMINATOR LOSS
        # interpolate in latent space with a randomly-chosen alpha
        alpha = tf.random_uniform([tf.shape(encode)[0], 1, 1, 1], 0, 1)
        alpha = 0.5 - tf.abs(alpha - 0.5)  # [0, 1] -> [0, 0.5]
        encode_mix = alpha * encode + (1 - alpha) * encode[::-1]
        decode_mix = decoder(encode_mix)

        loss_disc = tf.reduce_mean(
            tf.square(disc(decode_mix) - alpha[:, 0, 0, 0]))
        loss_disc_real = tf.reduce_mean(tf.square(disc(vae + reg * (x - vae))))
        # vae wants disc to predict 0
        loss_vae_disc = tf.reduce_mean(tf.square(disc(decode_mix)))
        utils.HookReport.log_tensor(loss_disc_real, 'loss_disc_real')

        # CLASSIFY (determine "usefulness" of latent codes)
        xops = classifiers.single_layer_classifier(
            tf.stop_gradient(encode), l, self.nclass)
        xloss = tf.reduce_mean(xops.loss)
        utils.HookReport.log_tensor(xloss, 'classify_latent')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        ae_vars = tf.global_variables('ae_')
        disc_vars = tf.global_variables('disc')
        xl_vars = tf.global_variables('single_layer_classifier')
        with tf.control_dependencies(update_ops):
            train_vae = tf.train.AdamOptimizer(FLAGS.lr).minimize(
                loss_vae + advweight * loss_vae_disc,
                var_list=ae_vars)
            train_d = tf.train.AdamOptimizer(FLAGS.lr).minimize(
                loss_disc + loss_disc_real,
                var_list=disc_vars)
            train_xl = tf.train.AdamOptimizer(FLAGS.lr).minimize(
                xloss, tf.train.get_global_step(), var_list=xl_vars)
        ops = train.AEOps(x, h, l, encode, decode, vae,
                          tf.group(train_vae, train_d, train_xl),
                          classify_latent=xops.output)

        n_interpolations = 16
        n_images_per_interpolation = 16

        def gen_images():
            return self.make_sample_grid_and_save(
                ops, interpolation=n_interpolations,
                height=n_images_per_interpolation)

        recon, inter, slerp, samples = tf.py_func(
            gen_images, [], [tf.float32] * 4)
        tf.summary.image('reconstruction', tf.expand_dims(recon, 0))
        tf.summary.image('interpolation', tf.expand_dims(inter, 0))
        tf.summary.image('slerp', tf.expand_dims(slerp, 0))
        tf.summary.image('samples', tf.expand_dims(samples, 0))

        if FLAGS.dataset == 'lines32':
            batched = (n_interpolations, 32, n_images_per_interpolation, 32, 1)
            batched_interp = tf.transpose(
                tf.reshape(inter, batched), [0, 2, 1, 3, 4])
            mean_distance, mean_smoothness = tf.py_func(
                eval.line_eval, [batched_interp], [tf.float32, tf.float32])
            tf.summary.scalar('mean_distance', mean_distance)
            tf.summary.scalar('mean_smoothness', mean_smoothness)

        return ops


def main(argv):
    del argv  # Unused.
    batch = FLAGS.batch
    dataset = data.get_dataset(FLAGS.dataset, dict(batch_size=batch))
    scales = int(round(math.log(dataset.width // FLAGS.latent_width, 2)))
    model = VAAIN(
        dataset,
        FLAGS.train_dir,
        latent=FLAGS.latent,
        depth=FLAGS.depth,
        scales=scales,
        beta=FLAGS.beta,
        advweight=FLAGS.advweight,
        advdepth=FLAGS.advdepth or FLAGS.depth,
        reg=FLAGS.reg)
    model.train()


if __name__ == '__main__':
    flags.DEFINE_integer('depth', 64, 'Depth (number of channels) before first convolution.')
    flags.DEFINE_integer(
        'latent', 16,
        'Latent space depth (number of channels). The total latent dimension is latent depth * '
        'latent_width ** 2.')
    flags.DEFINE_integer('latent_width', 4,
                         'Width of the latent space (width/height of an "image" in latent space)')
    flags.DEFINE_float('beta', 1.0, 'scale hyperparam for the ELBO KL term.')

    flags.DEFINE_float('advweight', 0.5, 'Adversarial weight (how much the VAE should care '
                                         'about fooling the discriminator).')
    flags.DEFINE_integer('advdepth', 0, 'Depth for adversary network.')
    flags.DEFINE_float('reg', 0.2, 'Amount of discriminator regularization.')

    app.run(main)
