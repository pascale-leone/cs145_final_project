import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

class VariationalAutoencoder(nn.Module):
    def __init__(
            self,
            q_sigma=0.05,
            n_dims_code=2,
            n_dims_data=400,
            hidden_layer_sizes=[32]):
        super(VariationalAutoencoder, self).__init__()
        self.n_dims_data = n_dims_data
        self.n_dims_code = n_dims_code
        self.q_sigma = torch.Tensor([float(q_sigma)])
        self.kwargs = dict(
            q_sigma=q_sigma, n_dims_code=n_dims_code,
            n_dims_data=n_dims_data, hidden_layer_sizes=hidden_layer_sizes)
        encoder_layer_sizes = (
            [n_dims_data] + hidden_layer_sizes + [n_dims_code]
            )
        self.n_layers = len(encoder_layer_sizes) - 1
        # Create the encoder, layer by layer
        self.encoder_activations = list()
        self.encoder_params = nn.ModuleList()
        for layer_id, (n_in, n_out) in enumerate(zip(
                encoder_layer_sizes[:-1], encoder_layer_sizes[1:])):
            self.encoder_params.append(nn.Linear(n_in, n_out))
            self.encoder_activations.append(F.relu)
        self.encoder_activations[-1] = lambda a: a

        self.decoder_activations = list()
        self.decoder_params = nn.ModuleList()
        decoder_layer_sizes = [a for a in reversed(encoder_layer_sizes)]
        for (n_in, n_out) in zip(
                decoder_layer_sizes[:-1], decoder_layer_sizes[1:]):
            self.decoder_params.append(nn.Linear(n_in, n_out))
            self.decoder_activations.append(F.relu)
        self.decoder_activations[-1] = torch.sigmoid

    def forward(self, x_ND):
        """ Run entire probabilistic autoencoder on input (encode then decode)

        Returns
        -------
        xproba_ND : 1D array, size of x_ND
        """
        mu_NC = self.encode(x_ND)
        z_NC = self.draw_sample_from_q(mu_NC)
        return self.decode(z_NC), mu_NC

    def draw_sample_from_q(self, mu_NC):
        ''' Draw sample from the probabilistic encoder q(z|mu(x), \sigma)

        We assume that "q" is Normal with:
        * mean mu (argument of this function)
        * stddev q_sigma (attribute of this class, use self.q_sigma)

        Args
        ----
        mu_NC : tensor-like, N x C
            Mean of the encoding for each of the N images in minibatch.

        Returns
        -------
        z_NC : tensor-like, N x C
            Exactly one sample vector for each of the N images in minibatch.
        '''
        N = mu_NC.shape[0]
        C = self.n_dims_code
        if self.training:
            # Draw standard normal samples "epsilon"
            eps_NC = torch.randn(N, C)
            # Using reparameterization trick,
            # Write a procedure here to make z_NC a valid draw from q 
            z_NC = self.q_sigma * eps_NC + mu_NC
            return z_NC
        else:
            # For evaluations, we always just use the mean
            return mu_NC

    def encode(self, x_ND):
        cur_arr = x_ND
        for ll in range(self.n_layers):
            linear_func = self.encoder_params[ll]
            a_func = self.encoder_activations[ll]
            cur_arr = a_func(linear_func(cur_arr))
        mu_NC = cur_arr
        return mu_NC

    def decode(self, z_NC):
        cur_arr = z_NC
        for ll in range(self.n_layers):
            linear_func = self.decoder_params[ll]
            a_func = self.decoder_activations[ll]
            cur_arr = a_func(linear_func(cur_arr))
        xproba_ND = cur_arr
        return xproba_ND

    def calc_vi_loss(self, x_ND, n_mc_samples=1, do_rescale=None):
        total_loss = 0.0
        mu_NC = self.encode(x_ND)

        N, D = x_ND.shape
        if do_rescale is None:
            Pbatch = 1.0
        else:
            Pbatch = N * D

        for ss in range(n_mc_samples):
            sample_z_NC = self.draw_sample_from_q(mu_NC)
            sample_xproba_ND = self.decode(sample_z_NC)
            sample_bce_loss = F.binary_cross_entropy(sample_xproba_ND, x_ND, reduction='sum')

            # KL divergence from q(mu, sigma) to prior (std normal)
            # see Appendix B from VAE paper
            # https://arxiv.org/pdf/1312.6114.pdf
            kl = -1/2 * torch.sum(1+torch.log(self.q_sigma.pow(2))-mu_NC.pow(2)-self.q_sigma.pow(2)) 
            total_loss += sample_bce_loss + kl
            print(f'kl: {kl}, sample_bce: {sample_bce_loss} total_loss: {total_loss}')

        return total_loss / float(Pbatch * n_mc_samples), sample_xproba_ND


    def train_for_one_epoch_of_gradient_update_steps(
            self, optimizer, train_loader, device, epoch, args):
        ''' Perform one epoch of gradient updates on provided model & data.

        Steps through dataset, one minibatch at a time.
        At each minibatch, we compute the gradient and step in that direction.

        Post Condition
        --------------
        This object's internal parameters are updated
        '''
        self.train() # mark as ready for training
        n_batch = len(train_loader)
        loss_per_batch = np.zeros(n_batch)
        num_batch_before_print = int(np.ceil(n_batch/5))

        for batch_idx, (batch_data, _) in enumerate(train_loader):
            # Reshape the data from 3dim to 2dim
            batch_x_ND = batch_data.to(device).view(-1, self.n_dims_data)
            
            # Zero out any stored gradients attached to the optimizer
            optimizer.zero_grad()

            # Compute the loss (and the required reconstruction as well)
            loss, batch_xproba_ND = self.calc_vi_loss(
                batch_x_ND,
                n_mc_samples=args.n_mc_samples,
                do_rescale=True)

            # Track loss over all batches
            loss_per_batch[batch_idx] = loss.item()

            # Compute the gradient of the loss wrt model parameters
            # (gradients are stored as attributes of parameters of 'model')
            loss.backward()

            # Take an optimization step (gradient descent step)
            optimizer.step() # side-effect: updates internals of self's model!

            # Done with this batch. Write a progress update to stdout, move on.
            is_last = batch_idx + 1 == len(train_loader)
            if (batch_idx + 1) % num_batch_before_print  == 0 or is_last:
                l1_dist = torch.mean(torch.abs(batch_x_ND - batch_xproba_ND))
                print("  epoch %3d | frac_seen %.2f | avg loss % .4f | batch loss % .4f | batch l1 % .3f" % (
                    epoch, (1+batch_idx) / float(len(train_loader)),
                    loss_per_batch[:(1+batch_idx)].mean(),
                    loss.item(),
                    l1_dist,
                    ))


    def save_to_file(self, fpath):
        """ Save this model to file
        """
        state_dict = self.state_dict()
        state_dict['kwargs'] = self.kwargs
        torch.save(state_dict, fpath)

    @classmethod
    def save_model_to_file(cls, model, fpath):
        """ Save given model to file (class method)
        """
        model.save_to_file(fpath)

    @classmethod
    def load_model_from_file(cls, fpath):
        """ Load from file (class method)

        Usage
        -----
        >>> VariationalAutoencoder.load_model_from_file('path/to/model.pytorch')
        """
        state_dict = torch.load(fpath)
        kwargs = state_dict.pop('kwargs')
        model = cls(**kwargs)
        assert 'kwargs' not in state_dict
        model.load_state_dict(state_dict)
        return model
