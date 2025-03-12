import torch
from torch import nn
import torch.nn.functional as F


class HeadDropout(nn.Module):
    def __init__(self, p=0.5):
        super(HeadDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p

    def forward(self, x):
        # If in evaluation mode, return the input as-is
        if not self.training:
            return x
        
        # Create a binary mask of the same shape as x
        binary_mask = (torch.rand_like(x) > self.p).float()
        
        # Set dropped values to negative infinity during training
        return x * binary_mask + (1 - binary_mask) * -1e20

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DETM(nn.Module):
    """
    Dynamic Embedded Topic Model with Decomposed Alpha Trajectories (Trend + Seasonal) and MoLE for Seasonal Component,
    using a SINGLE AR(1) PRIOR for the COMBINED Alpha.
    """
    def __init__(self, vocab_size, num_times, train_size, train_time_wordfreq, num_topics=200, train_WE=True, pretrained_WE=None, en_units=800, eta_hidden_size=200, rho_size=300, enc_drop=0.0, eta_nlayers=3, eta_dropout=0.0, delta=0.005, theta_act='relu', device='cpu',
                 num_seasonal_experts=3, alpha_mixing_units=100, decomp_kernel_size=25, head_dropout=0.0):
        super().__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.num_times = num_times
        self.vocab_size = vocab_size
        self.eta_hidden_size = eta_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.eta_nlayers = eta_nlayers
        self.t_drop = nn.Dropout(enc_drop)
        self.eta_dropout = eta_dropout
        self.delta = delta
        self.train_WE = train_WE
        self.train_size = train_size
        self.rnn_inp = train_time_wordfreq
        self.device = device
        self.theta_act = self.get_activation(theta_act)

        ## Decomposition related
        self.decomp_kernel_size = decomp_kernel_size
        self.decomposition = series_decomp(decomp_kernel_size) # Decomposition block

        ## MoLE related parameters for SEASONAL Alpha Component
        self.num_seasonal_experts = num_seasonal_experts
        self.alpha_mixing_units = alpha_mixing_units
        self.head_dropout = HeadDropout(head_dropout)

        ## define the word embedding matrix \rho
        if self.train_WE:
            self.rho = nn.Linear(self.rho_size, self.vocab_size, bias=False)
        else:
            rho = nn.Embedding(pretrained_WE.size())
            rho.weight.data = torch.from_numpy(pretrained_WE)
            self.rho = rho.weight.data.clone().float().to(self.device)

        ## Variational parameters for TREND component of alpha (Directly parameterized)
        self.mu_q_alpha_trend = nn.Parameter(torch.randn(self.num_topics, self.num_times, self.rho_size))
        self.logsigma_q_alpha_trend = nn.Parameter(torch.randn(self.num_topics, self.num_times, self.rho_size))

        ## Variational parameters for SEASONAL component of alpha (MoLE for seasonal component)
        self.mu_q_alpha_seasonal_experts = nn.Parameter(torch.randn(self.num_seasonal_experts, self.num_topics, self.num_times, self.rho_size))
        self.logsigma_q_alpha_seasonal_experts = nn.Parameter(torch.randn(self.num_seasonal_experts, self.num_topics, self.num_times, self.rho_size))

        ## Mixing network for SEASONAL Alpha Component (MoLE for seasonal)
        self.alpha_mixing_net_seasonal = nn.Sequential(
            nn.Linear(1, self.alpha_mixing_units),  # Input is time step index
            nn.ReLU(),
            nn.Linear(self.alpha_mixing_units, self.num_seasonal_experts),
            self.head_dropout,
            nn.Softmax(dim=-1)  # Output weights for seasonal experts
        )

        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
            nn.Linear(self.vocab_size + self.num_topics, en_units),
            self.theta_act,
            nn.Linear(en_units, en_units),
            self.theta_act,
        )
        self.mu_q_theta = nn.Linear(en_units, self.num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(en_units, self.num_topics, bias=True)

        ## define variational distribution for \eta via amortizartion
        self.q_eta_map = nn.Linear(self.vocab_size, self.eta_hidden_size)
        self.q_eta = nn.LSTM(self.eta_hidden_size, self.eta_hidden_size, self.eta_nlayers, dropout=self.eta_dropout)
        self.mu_q_eta = nn.Linear(self.eta_hidden_size + self.num_topics, self.num_topics, bias=True)
        self.logsigma_q_eta = nn.Linear(self.eta_hidden_size + self.num_topics, self.num_topics, bias=True)

        self.decoder_bn = nn.BatchNorm1d(vocab_size)
        self.decoder_bn.weight.requires_grad = False
    def get_activation(self, act):
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'softplus': nn.Softplus(),
            'rrelu': nn.RReLU(),
            'leakyrelu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'glu': nn.GLU(),
        }

        if act in activations:
            act = activations[act]
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        """Returns KL( N(q_mu, q_logsigma) || N(p_mu, p_logsigma) ).
        """
        if p_mu is not None and p_logsigma is not None:
            sigma_q_sq = torch.exp(q_logsigma)
            sigma_p_sq = torch.exp(p_logsigma)
            kl = ( sigma_q_sq + (q_mu - p_mu)**2 ) / ( sigma_p_sq + 1e-6 )
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1)
        else:
            kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1)
        return kl
    
    def get_alpha(self): ## Decomposed MoLE version of get_alpha - SINGLE PRIOR for COMBINED Alpha
        alphas = torch.zeros(self.num_times, self.num_topics, self.rho_size).to(self.device)
        kl_alpha = []

        trend_component = torch.zeros(self.num_times, self.num_topics, self.rho_size).to(self.device)
        seasonal_component = torch.zeros(self.num_times, self.num_topics, self.rho_size).to(self.device)

        # Prior parameters for COMBINED ALPHA (AR(1) prior - as in original DETM)
        p_mu_alpha = torch.zeros(self.num_topics, self.rho_size).to(self.device) # Initial prior mean for alpha
        logsigma_p_alpha = torch.zeros(self.num_topics, self.rho_size).to(self.device)
        logsigma_delta_alpha = torch.log(self.delta * torch.ones(self.num_topics, self.rho_size).to(self.device)) # Delta for alpha prior

        prev_alpha = None # To track previous timestep's alpha for AR(1) prior

        for t in range(self.num_times): 
            # 1. TREND COMPONENT (Directly Parameterized - No MoLE)
            mu_trend_t = self.mu_q_alpha_trend[:, t, :]
            logsigma_trend_t = self.logsigma_q_alpha_trend[:, t, :]
            trend_sample_t = self.reparameterize(mu_trend_t, logsigma_trend_t)
            trend_component[t] = trend_sample_t

            # 2. SEASONAL COMPONENT (MoLE for Seasonal)
            mixing_weights_seasonal = self.alpha_mixing_net_seasonal(torch.tensor([[t/self.num_times]]).float().to(self.device)) # Mixing weights for seasonal experts

            seasonal_experts_samples_t = []
            seasonal_component_kl_t = [] # KL for seasonal will be against standard normal (or simpler prior)

            for expert_idx in range(self.num_seasonal_experts):
                mu_seasonal_expert_t = self.mu_q_alpha_seasonal_experts[expert_idx, :, t, :]
                logsigma_seasonal_expert_t = self.logsigma_q_alpha_seasonal_experts[expert_idx, :, t, :]

                seasonal_expert_sample_t = self.reparameterize(mu_seasonal_expert_t, logsigma_seasonal_expert_t)
                seasonal_experts_samples_t.append(seasonal_expert_sample_t)

                # KL Divergence for SEASONAL expert component (against standard normal prior) - SIMPLER PRIOR for SEASONAL
                p_mu_seasonal_component = torch.zeros(self.num_topics, self.rho_size).to(self.device) # Standard normal prior for seasonal component
                logsigma_p_seasonal_component = torch.zeros(self.num_topics, self.rho_size).to(self.device)
                kl_seasonal_expert_t = self.get_kl(mu_seasonal_expert_t, logsigma_seasonal_expert_t, p_mu_seasonal_component, logsigma_p_seasonal_component)
                seasonal_component_kl_t.append(kl_seasonal_expert_t)

            # Combine seasonal experts using mixing weights
            mixed_seasonal_t = torch.zeros(self.num_topics, self.rho_size).to(self.device)
            for expert_idx in range(self.num_seasonal_experts):
                mixed_seasonal_t += mixing_weights_seasonal[0, expert_idx] * seasonal_experts_samples_t[expert_idx]
            seasonal_component[t] = mixed_seasonal_t

            # Expected KL for SEASONAL component (MoLE)
            expected_kl_seasonal_t = torch.zeros_like(seasonal_component_kl_t[0]).to(self.device)
            for expert_idx in range(self.num_seasonal_experts):
                expected_kl_seasonal_t += mixing_weights_seasonal[0, expert_idx] * seasonal_component_kl_t[expert_idx]
            kl_alpha.append(expected_kl_seasonal_t)

            # 3. RECOMBINE TREND and SEASONAL to get final alpha
            alphas[t] = trend_component[t] + seasonal_component[t]

            # KL Divergence for TREND component (using AR(1) prior for COMBINED ALPHA) - APPROXIMATION
            # We are approximating the prior for TREND as being similar to the prior for COMBINED ALPHA
            if t == 0:
                kl_trend_t = self.get_kl(mu_trend_t, logsigma_trend_t, p_mu_alpha, logsigma_p_alpha) # KL against initial prior of COMBINED ALPHA
            else:
                p_mu_alpha = prev_alpha # AR(1) prior for COMBINED ALPHA - using previous *combined* alpha as prior mean
                kl_trend_t = self.get_kl(mu_trend_t, logsigma_trend_t, p_mu_alpha, logsigma_delta_alpha) # KL against AR(1) prior of COMBINED ALPHA
            kl_alpha.append(kl_trend_t)

            prev_alpha = alphas[t] # Update previous alpha for AR(1) prior - using the *combined* alpha

        kl_alpha = torch.stack(kl_alpha).sum() # Sum KL terms for both trend and seasonal components
        return alphas, kl_alpha.sum()
    def get_eta(self, rnn_inp): ## structured amortized inference
        inp = self.q_eta_map(rnn_inp).unsqueeze(1)
        hidden = self.init_hidden()
        output, _ = self.q_eta(inp, hidden)
        output = output.squeeze()

        etas = torch.zeros(self.num_times, self.num_topics).to(self.device)
        kl_eta = []

        inp_0 = torch.cat([output[0], torch.zeros(self.num_topics,).to(self.device)], dim=0)
        mu_0 = self.mu_q_eta(inp_0)
        logsigma_0 = self.logsigma_q_eta(inp_0)
        etas[0] = self.reparameterize(mu_0, logsigma_0)

        p_mu_0 = torch.zeros(self.num_topics,).to(self.device)
        logsigma_p_0 = torch.zeros(self.num_topics,).to(self.device)
        kl_0 = self.get_kl(mu_0, logsigma_0, p_mu_0, logsigma_p_0)
        kl_eta.append(kl_0)

        for t in range(1, self.num_times):
            inp_t = torch.cat([output[t], etas[t-1]], dim=0)
            mu_t = self.mu_q_eta(inp_t)
            logsigma_t = self.logsigma_q_eta(inp_t)
            etas[t] = self.reparameterize(mu_t, logsigma_t)

            p_mu_t = etas[t-1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics,).to(self.device))
            kl_t = self.get_kl(mu_t, logsigma_t, p_mu_t, logsigma_p_t)
            kl_eta.append(kl_t)
        kl_eta = torch.stack(kl_eta).sum()

        return etas, kl_eta

    def get_theta(self, bows, times, eta=None): ## amortized inference
        """Returns the topic proportions.
        """
        normalized_bows = bows / bows.sum(1, keepdims=True)

        if eta is None and self.training is False:
            eta, kl_eta = self.get_eta(self.rnn_inp)

        eta_td = eta[times]
        inp = torch.cat([normalized_bows, eta_td], dim=1)
        q_theta = self.q_theta(inp)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        kl_theta = self.get_kl(mu_theta, logsigma_theta, eta_td, torch.zeros(self.num_topics).to(self.device))

        if self.training:
            return theta, kl_theta
        else:
            return theta

    @property
    def word_embeddings(self):
        return self.rho.weight

    @property
    def topic_embeddings(self):
        alpha, _ = self.get_alpha()
        return alpha

    def get_beta(self, alpha=None):
        """Returns the topic matrix \beta of shape T x K x V
        """
        if alpha is None and self.training is False:
            alpha, kl_alpha = self.get_alpha() # get_alpha now returns samples from MoLE

        if self.train_WE:
            logit = self.rho(alpha.view(alpha.size(0) * alpha.size(1), self.rho_size))
        else:
            tmp = alpha.view(alpha.size(0) * alpha.size(1), self.rho_size)
            logit = torch.mm(tmp, self.rho.permute(1, 0))
        logit = logit.view(alpha.size(0), alpha.size(1), -1)

        beta = F.softmax(logit, dim=-1)

        return beta

    def get_NLL(self, theta, beta, bows):
        theta = theta.unsqueeze(1)
        loglik = torch.bmm(theta, beta).squeeze(1)
        loglik = torch.log(loglik + 1e-12)
        nll = -loglik * bows
        nll = nll.sum(-1)
        return nll

    def forward(self, bows, times):
        bsz = bows.size(0)
        coeff = self.train_size / bsz
        eta, kl_eta = self.get_eta(self.rnn_inp)
        theta, kl_theta = self.get_theta(bows, times, eta)
        kl_theta = kl_theta.sum() * coeff

        alpha, kl_alpha = self.get_alpha()
        beta = self.get_beta(alpha)

        beta = beta[times]
        nll = self.get_NLL(theta, beta, bows)
        nll = nll.sum() * coeff

        loss = nll + kl_eta + kl_theta

        rst_dict = {
            'loss': loss,
            'nll': nll,
            'kl_eta': kl_eta,
            'kl_theta': kl_theta
        }

        loss += kl_alpha
        rst_dict['kl_alpha'] = kl_alpha

        return rst_dict

    def init_hidden(self):
        """Initializes the first hidden state of the RNN used as inference network for \\eta.
        """
        weight = next(self.parameters())
        nlayers = self.eta_nlayers
        nhid = self.eta_hidden_size
        print('hahah')
        return (weight.new_zeros(nlayers, 1, nhid), weight.new_zeros(nlayers, 1, nhid))
