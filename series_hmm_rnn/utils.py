import torch, torch.nn.functional as F

def build_Q_seq(actions, rewards, transitions, agents):
    B, T, device = actions.size(0), actions.size(1), actions.device
    Qs = []
    for _, agent in agents:
        if hasattr(agent, "forward"):
            # route by arity
            try:
                Qs.append(agent.forward(actions, rewards, T))
            except TypeError:
                try:
                    Qs.append(agent.forward(actions, T))
                except TypeError:
                    try:
                        Qs.append(agent.forward(actions, rewards, transitions, T))
                    except TypeError:
                        Qs.append(agent.forward(B, T, device))
        else:
            Qs.append(torch.zeros(B, T, 2, device=device))
    return torch.stack(Qs, dim=2)  # [B,T,A,2]

def make_inputs(actions, rewards, transitions):
    return torch.stack([actions.float(), rewards.float(), transitions.float()], dim=-1)

def nll_loss(pi_log, actions):
    idx = actions.long().unsqueeze(-1)
    return -(pi_log.gather(-1, idx).squeeze(-1)).mean()
