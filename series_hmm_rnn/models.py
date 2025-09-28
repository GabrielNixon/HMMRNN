import math, torch, torch.nn as nn, torch.nn.functional as F

class TinyMoARNN(nn.Module):
    def __init__(self, n_agents=4, hidden=2):
        super().__init__()
        self.rnn = nn.GRU(input_size=3, hidden_size=hidden, batch_first=True)
        self.Wg  = nn.Linear(hidden, n_agents)
        self.by  = nn.Parameter(torch.zeros(2))
    def forward(self, x, Q_seq):
        h0 = torch.zeros(1, x.size(0), self.rnn.hidden_size, device=x.device)
        h, _ = self.rnn(x, h0)
        g = F.softmax(self.Wg(h), dim=-1)                 # [B,T,A]
        V = torch.einsum('bta,btay->bty', g, Q_seq)       # [B,T,2]
        V = V + self.by.view(1,1,2)
        return F.log_softmax(V, dim=-1), g

class DiscreteHMM(nn.Module):
    def __init__(self, K=2):
        super().__init__()
        self.log_pi = nn.Parameter(torch.zeros(K))
        self.log_A  = nn.Parameter(torch.zeros(K, K))
    def forward_backward(self, emission_logp):            # [B,T,K]
        B, T, K = emission_logp.shape
        log_pi = F.log_softmax(self.log_pi, dim=0)
        log_A  = F.log_softmax(self.log_A, dim=1)
        alpha = emission_logp.new_empty(B, T, K)
        alpha[:, 0] = log_pi + emission_logp[:, 0]
        for t in range(1, T):
            m = alpha[:, t-1].unsqueeze(-1) + log_A.unsqueeze(0)
            alpha[:, t] = torch.logsumexp(m, dim=1) + emission_logp[:, t]
        beta = emission_logp.new_zeros(B, T, K)
        for t in range(T-2, -1, -1):
            m = beta[:, t+1].unsqueeze(1) + log_A.unsqueeze(0) + emission_logp[:, t+1].unsqueeze(1)
            beta[:, t] = torch.logsumexp(m, dim=2)
        log_gamma = alpha + beta
        return log_gamma - torch.logsumexp(log_gamma, dim=-1, keepdim=True)

class SeriesHMMTinyMoARNN(nn.Module):
    """Likelihood-coupled HMM over per-phase TinyMoA heads."""
    def __init__(self, n_agents=4, hidden=6, K=2, tau=1.25):
        super().__init__()
        self.K = K; self.tau = tau
        self.rnn = nn.GRU(input_size=3, hidden_size=hidden, batch_first=True)
        self.Wg  = nn.Parameter(torch.zeros(K, hidden, n_agents))  # per-phase gates
        self.by  = nn.Parameter(torch.zeros(K, 2))                 # per-phase action bias
        nn.init.xavier_uniform_(self.Wg)
        self.hmm = DiscreteHMM(K)
    def _per_phase_logits(self, h, Q_seq):
        Wh = torch.einsum('bth,khA->btkA', h, self.Wg)  # [B,T,K,A]
        gk = torch.softmax(Wh, dim=-1)                  # [B,T,K,A]
        Vk = torch.einsum('btkA,btAy->btky', gk, Q_seq) # [B,T,K,2]
        return Vk + self.by.view(1,1,self.K,2), gk
    def forward(self, x, Q_seq, actions=None):
        B = x.size(0)
        h0 = torch.zeros(1, B, self.rnn.hidden_size, device=x.device)
        h, _ = self.rnn(x, h0)
        Vk, gk = self._per_phase_logits(h, Q_seq)             # [B,T,K,2], [B,T,K,A]
        log_pi_k = F.log_softmax(Vk, dim=-1)                  # [B,T,K,2]
        if actions is None:
            log_gamma = x.new_zeros(B, x.size(1), self.K) - math.log(self.K)
        else:
            idx  = actions.long().unsqueeze(-1).unsqueeze(-1).expand(-1,-1,self.K,1)
            emis = log_pi_k.gather(-1, idx).squeeze(-1) * self.tau
            log_gamma = self.hmm.forward_backward(emis)       # [B,T,K]
        log_pi_marg = torch.logsumexp(log_gamma.unsqueeze(-1) + log_pi_k, dim=2)   # [B,T,2]
        return log_pi_marg, gk, log_gamma
