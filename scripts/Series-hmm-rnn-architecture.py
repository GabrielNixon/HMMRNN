import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyMoARNN(nn.Module):
    def __init__(self, n_agents=5, hidden=2):
        super().__init__()
        self.rnn = nn.GRU(input_size=3, hidden_size=hidden, batch_first=True)
        self.Wg = nn.Linear(hidden, n_agents)
        self.by = nn.Parameter(torch.zeros(2))

    def forward(self, x, Q_seq):
        h0 = torch.zeros(1, x.size(0), self.rnn.hidden_size, device=x.device)
        h, _ = self.rnn(x, h0)
        g = F.softmax(self.Wg(h), dim=-1)
        V = torch.einsum('bta,btay->bty', g, Q_seq)
        V = V + self.by
        pi_log = F.log_softmax(V, dim=-1)
        return pi_log, g

def nll_loss(pi_log, actions):
    idx = actions.long().unsqueeze(-1)
    return -(pi_log.gather(-1, idx).squeeze(-1)).mean()

class MFReward:
    def __init__(self, alpha=0.2, decay=0.0):
        self.alpha = alpha
        self.decay = decay
    def forward(self, actions, rewards, T):
        B = actions.size(0)
        Q = torch.zeros(B, 2, device=actions.device)
        out = []
        for t in range(T):
            out.append(Q.unsqueeze(1))
            a = actions[:, t]
            r = rewards[:, t].float()
            Q = (1 - self.decay) * Q
            Qa = Q.gather(1, a.unsqueeze(1)).squeeze(1)
            Qa = Qa + self.alpha * (r - Qa)
            Q = Q.scatter(1, a.unsqueeze(1), Qa.unsqueeze(1))
        return torch.cat(out, dim=1)

class MFChoice:
    def __init__(self, kappa=0.1, rho=0.0):
        self.kappa = kappa
        self.rho = rho
    def forward(self, actions, T):
        B = actions.size(0)
        Q = torch.zeros(B, 2, device=actions.device)
        out = []
        for t in range(T):
            out.append(Q.unsqueeze(1))
            a = actions[:, t]
            Q = (1 - self.rho) * Q
            Qa = Q.gather(1, a.unsqueeze(1)).squeeze(1) + self.kappa
            Q = Q.scatter(1, a.unsqueeze(1), Qa.unsqueeze(1))
        return torch.cat(out, dim=1)

class BiasAgent:
    def __init__(self, bias_left=0.0, bias_right=0.0):
        self.bias = torch.tensor([bias_left, bias_right])
    def forward(self, B, T, device):
        return self.bias.to(device).view(1, 1, 2).repeat(B, T, 1)

class MBCommonRareStub:
    def __init__(self):
        pass
    def forward(self, B, T, device):
        return torch.zeros(B, T, 2, device=device)

def build_Q_seq(actions, rewards, transitions, agents):
    B, T = actions.size(0), actions.size(1)
    device = actions.device
    Qs = []
    for name, agent in agents:
        if isinstance(agent, MFReward):
            Qs.append(agent.forward(actions, rewards, T))
        elif isinstance(agent, MFChoice):
            Qs.append(agent.forward(actions, T))
        elif isinstance(agent, BiasAgent):
            Qs.append(agent.forward(B, T, device))
        elif isinstance(agent, MBCommonRareStub):
            Qs.append(agent.forward(B, T, device))
        else:
            Qs.append(torch.zeros(B, T, 2, device=device))
    Q_seq = torch.stack(Qs, dim=2)
    return Q_seq

def make_inputs(actions, rewards, transitions):
    x = torch.stack([actions.float(), rewards.float(), transitions.float()], dim=-1)
    return x

def train_epoch(model, opt, actions, rewards, transitions, agents):
    model.train()
    x = make_inputs(actions, rewards, transitions)
    Q_seq = build_Q_seq(actions, rewards, transitions, agents)
    pi_log, g = model(x, Q_seq)
    loss = nll_loss(pi_log, actions)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        acc = (pi_log.argmax(-1) == actions).float().mean().item()
        g_mean = g.mean(dim=(0, 1)).detach().cpu()
    return loss.item(), acc, g_mean

def eval_epoch(model, actions, rewards, transitions, agents):
    model.eval()
    with torch.no_grad():
        x = make_inputs(actions, rewards, transitions)
        Q_seq = build_Q_seq(actions, rewards, transitions, agents)
        pi_log, g = model(x, Q_seq)
        loss = nll_loss(pi_log, actions)
        acc = (pi_log.argmax(-1) == actions).float().mean().item()
        g_mean = g.mean(dim=(0, 1)).detach().cpu()
    return loss.item(), acc, g_mean

def synthetic_batch(B=32, T=100, p_reward=0.6, seed=0, device='cpu'):
    g = torch.Generator().manual_seed(seed)
    actions = torch.randint(0, 2, (B, T), generator=g, device=device)
    rewards = torch.bernoulli(torch.full((B, T), p_reward, device=device))
    transitions = torch.bernoulli(torch.full((B, T), 0.7, device=device))
    return actions.long(), rewards.long(), transitions.long()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agents = [
        ('Model-free value', MFReward(alpha=0.3, decay=0.0)),
        ('Model-free choice', MFChoice(kappa=0.1, rho=0.0)),
        ('Model-based common', MBCommonRareStub()),
        ('Model-based rare', MBCommonRareStub()),
        ('Bias', BiasAgent(0.0, 0.0)),
    ]
    model = TinyMoARNN(n_agents=len(agents), hidden=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    a_tr, r_tr, t_tr = synthetic_batch(B=64, T=200, seed=1, device=device)
    a_te, r_te, t_te = synthetic_batch(B=64, T=200, seed=2, device=device)
    for epoch in range(10):
        loss, acc, gmean = train_epoch(model, opt, a_tr, r_tr, t_tr, agents)
        val_loss, val_acc, vgmean = eval_epoch(model, a_te, r_te, t_te, agents)
    with torch.no_grad():
        x = make_inputs(a_te, r_te, t_te)
        Q_seq = build_Q_seq(a_te, r_te, t_te, agents)
        _, g = model(x, Q_seq)
        torch.save({'model': model.state_dict(), 'g_mean': vgmean, 'g_seq': g.cpu()}, 'tiny_moa_rnn_ckpt.pt')
        torch.save({'actions': a_te.cpu(), 'rewards': r_te.cpu(), 'transitions': t_te.cpu()}, 'tiny_moa_rnn_eval_batch.pt')


class PhaseEmitter(nn.Module):
    def __init__(self, in_dim=3, hidden=8, K=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, K)
    def forward(self, x):
        B, T, D = x.shape
        z = torch.tanh(self.fc1(x))
        e = F.log_softmax(self.fc2(z), dim=-1)
        return e

class DiscreteHMM(nn.Module):
    def __init__(self, K=2):
        super().__init__()
        self.log_pi = nn.Parameter(torch.zeros(K))
        self.log_A = nn.Parameter(torch.zeros(K, K))
    def forward(self, emission_logp):
        B, T, K = emission_logp.shape
        log_pi = F.log_softmax(self.log_pi, dim=0)
        log_A = F.log_softmax(self.log_A, dim=1)
        alpha = emission_logp.new_zeros(B, T, K)
        alpha[:, 0] = log_pi + emission_logp[:, 0]
        for t in range(1, T):
            prev = alpha[:, t-1].unsqueeze(-1) + log_A.unsqueeze(0)
            alpha[:, t] = torch.logsumexp(prev, dim=1) + emission_logp[:, t]
        beta = emission_logp.new_zeros(B, T, K)
        for t in range(T-2, -1, -1):
            nxt = beta[:, t+1].unsqueeze(1) + log_A.unsqueeze(0) + emission_logp[:, t+1].unsqueeze(1)
            beta[:, t] = torch.logsumexp(nxt, dim=2)
        log_gamma = alpha + beta
        log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=-1, keepdim=True)
        return log_gamma

class SeriesHMMTinyMoARNN(nn.Module):
    def __init__(self, n_agents=5, hidden=2, K=2, emit_hidden=8):
        super().__init__()
        self.K = K
        self.rnn = nn.GRU(input_size=3, hidden_size=hidden, batch_first=True)
        self.Wg = nn.Parameter(torch.zeros(K, hidden, n_agents))
        self.by = nn.Parameter(torch.zeros(K, 2))
        self.emitter = PhaseEmitter(3, emit_hidden, K)
        self.hmm = DiscreteHMM(K)
    def forward(self, x, Q_seq):
        h0 = torch.zeros(1, x.size(0), self.rnn.hidden_size, device=x.device)
        h, _ = self.rnn(x, h0)
        e = self.emitter(x)
        log_gamma = self.hmm(e)
        gamma = torch.exp(log_gamma)
        Wh = torch.einsum('bth,khA->btkA', h, self.Wg)
        gk = F.softmax(Wh, dim=-1)
        g = torch.einsum('btk,btkA->btA', gamma, gk)
        V = torch.einsum('bta,btay->bty', g, Q_seq)
        by = torch.einsum('btk,kc->btc', gamma, self.by)
        V = V + by
        pi_log = F.log_softmax(V, dim=-1)
        return pi_log, g, log_gamma

def train_epoch_hmm(model, opt, actions, rewards, transitions, agents):
    model.train()
    x = make_inputs(actions, rewards, transitions)
    Q_seq = build_Q_seq(actions, rewards, transitions, agents)
    pi_log, g, lg = model(x, Q_seq)
    loss = nll_loss(pi_log, actions)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        acc = (pi_log.argmax(-1) == actions).float().mean().item()
        g_mean = g.mean(dim=(0, 1)).detach().cpu()
    return loss.item(), acc, g_mean

def eval_epoch_hmm(model, actions, rewards, transitions, agents):
    model.eval()
    with torch.no_grad():
        x = make_inputs(actions, rewards, transitions)
        Q_seq = build_Q_seq(actions, rewards, transitions, agents)
        pi_log, g, lg = model(x, Q_seq)
        loss = nll_loss(pi_log, actions)
        acc = (pi_log.argmax(-1) == actions).float().mean().item()
        g_mean = g.mean(dim=(0, 1)).detach().cpu()
    return loss.item(), acc, g_mean

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agents = [
        ('Model-free value', MFReward(alpha=0.3, decay=0.0)),
        ('Model-free choice', MFChoice(kappa=0.1, rho=0.0)),
        ('Model-based common', MBCommonRareStub()),
        ('Model-based rare', MBCommonRareStub()),
        ('Bias', BiasAgent(0.0, 0.0)),
    ]
    K = 2
    model = SeriesHMMTinyMoARNN(n_agents=len(agents), hidden=2, K=K, emit_hidden=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    a_tr, r_tr, t_tr = synthetic_batch(B=64, T=200, seed=3, device=device)
    a_te, r_te, t_te = synthetic_batch(B=64, T=200, seed=4, device=device)
    for epoch in range(10):
        loss, acc, gmean = train_epoch_hmm(model, opt, a_tr, r_tr, t_tr, agents)
        val_loss, val_acc, vgmean = eval_epoch_hmm(model, a_te, r_te, t_te, agents)
    with torch.no_grad():
        x = make_inputs(a_te, r_te, t_te)
        Q_seq = build_Q_seq(a_te, r_te, t_te, agents)
        pi_log, g, lg = model(x, Q_seq)
        torch.save({'model': model.state_dict(), 'g_mean': vgmean, 'g_seq': g.cpu()}, 'series_hmm_tinymoa_ckpt.pt')
        torch.save({'actions': a_te.cpu(), 'rewards': r_te.cpu(), 'transitions': t_te.cpu()}, 'series_hmm_eval_batch.pt')

# scripts/train_dual.py
import argparse, torch

def train_dual_main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=['tiny','series_hmm'], default='tiny')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--T', type=int, default=200)
    p.add_argument('--hidden', type=int, default=2)
    p.add_argument('--K', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default='ckpt')
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    agents = [
        ('Model-free value', MFReward(alpha=0.3, decay=0.0)),
        ('Model-free choice', MFChoice(kappa=0.1, rho=0.0)),
        ('Model-based common', MBCommonRareStub()),
        ('Model-based rare', MBCommonRareStub()),
        ('Bias', BiasAgent(0.0, 0.0)),
    ]
    if args.arch == 'tiny':
        model = TinyMoARNN(n_agents=len(agents), hidden=args.hidden).to(device)
    else:
        model = SeriesHMMTinyMoARNN(n_agents=len(agents), hidden=args.hidden, K=args.K, emit_hidden=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    a_tr, r_tr, t_tr = synthetic_batch(B=args.batch, T=args.T, seed=args.seed+1, device=device)
    a_te, r_te, t_te = synthetic_batch(B=args.batch, T=args.T, seed=args.seed+2, device=device)
    for _ in range(args.epochs):
        if args.arch == 'tiny':
            loss, acc, gmean = train_epoch(model, opt, a_tr, r_tr, t_tr, agents)
            vloss, vacc, vgmean = eval_epoch(model, a_te, r_te, t_te, agents)
        else:
            loss, acc, gmean = train_epoch_hmm(model, opt, a_tr, r_tr, t_tr, agents)
            vloss, vacc, vgmean = eval_epoch_hmm(model, a_te, r_te, t_te, agents)
    with torch.no_grad():
        x = make_inputs(a_te, r_te, t_te)
        Q_seq = build_Q_seq(a_te, r_te, t_te, agents)
        if args.arch == 'tiny':
            pi_log, g = model(x, Q_seq)
            torch.save({'arch': args.arch, 'model': model.state_dict(), 'g_mean': vgmean, 'g_seq': g.cpu()}, f'{args.out}_{args.arch}.pt')
        else:
            pi_log, g, lg = model(x, Q_seq)
            torch.save({'arch': args.arch, 'model': model.state_dict(), 'g_mean': vgmean, 'g_seq': g.cpu()}, f'{args.out}_{args.arch}.pt')

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and (sys.argv[1].startswith('--arch') or sys.argv[1] in ['--epochs','--batch','--T','--hidden','--K','--lr','--seed','--out']):
        train_dual_main()
