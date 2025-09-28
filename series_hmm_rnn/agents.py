import torch

class MFReward:
    def __init__(self, alpha=0.3, decay=0.0):
        self.alpha = alpha; self.decay = decay
    def forward(self, actions, rewards, T):
        B, device = actions.size(0), actions.device
        Q = torch.zeros(B, 2, device=device); out = []
        for t in range(T):
            out.append(Q.unsqueeze(1))
            a = actions[:, t]; r = rewards[:, t].float()
            Q = (1 - self.decay) * Q
            Qa = Q.gather(1, a.unsqueeze(1)).squeeze(1)
            Qa = Qa + self.alpha * (r - Qa)
            Q = Q.scatter(1, a.unsqueeze(1), Qa.unsqueeze(1))
        return torch.cat(out, dim=1)

class MFChoice:
    def __init__(self, kappa=0.2, rho=0.0):
        self.kappa = kappa; self.rho = rho
    def forward(self, actions, T):
        B, device = actions.size(0), actions.device
        Q = torch.zeros(B, 2, device=device); out = []
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
        return self.bias.to(device).view(1,1,2).repeat(B,T,1)

class MBReward:
    def __init__(self, p_common=0.7, alpha_state=0.2):
        self.p_common = p_common; self.alpha_state = alpha_state
    def forward(self, actions, rewards, transitions, T):
        B, device = actions.size(0), actions.device
        V = torch.zeros(B, 2, device=device); out = []
        for t in range(T):
            q0 = self.p_common*V[:,0] + (1-self.p_common)*V[:,1]
            q1 = self.p_common*V[:,1] + (1-self.p_common)*V[:,0]
            out.append(torch.stack([q0, q1], dim=1).unsqueeze(1))
            a = actions[:, t]; r = rewards[:, t].float(); tran = transitions[:, t]
            s_obs = torch.where(((a==0)&(tran==1))|((a==1)&(tran==0)), 0, 1).long()
            idx = s_obs.unsqueeze(1)
            v_sel = V.gather(1, idx).squeeze(1)
            v_upd = v_sel + self.alpha_state*(r - v_sel)
            V = V.scatter(1, idx, v_upd.unsqueeze(1))
        return torch.cat(out, dim=1)
