import torch

def two_step_mb_generator(B=64, T=400, p_common=0.7, beta=3.5, dwell=120, seed=1, device='cpu'):
    torch.manual_seed(seed)
    s = torch.zeros(B, T, dtype=torch.long, device=device)
    for b in range(B):
        t = 0; cur = torch.randint(0,2,(1,), device=device).item()
        while t < T:
            L = min(dwell + torch.randint(-10,11,(1,),device=device).item(), T-t)
            s[b, t:t+L] = cur; cur ^= 1; t += L
    R0 = torch.tensor([0.9, 0.1], device=device).view(1,1,2)
    R1 = torch.tensor([0.1, 0.9], device=device).view(1,1,2)
    R = torch.where(s.unsqueeze(-1)==0, R0, R1)    # [B,T,2]
    Qtrue = torch.empty(B, T, 2, device=device)
    Qtrue[:,:,0] = p_common*R[:,:,0] + (1-p_common)*R[:,:,1]
    Qtrue[:,:,1] = p_common*R[:,:,1] + (1-p_common)*R[:,:,0]
    pi = torch.softmax(beta*Qtrue, dim=-1)
    a = torch.distributions.Categorical(pi).sample()
    rare = torch.bernoulli(torch.full((B,T), 1-p_common, device=device)).long()
    s2 = torch.where(((a==0)&(rare==0))|((a==1)&(rare==1)), 0, 1)
    tran = (rare==0).long()
    b_ix = torch.arange(B, device=device).unsqueeze(1).expand(B,T)
    t_ix = torch.arange(T, device=device).unsqueeze(0).expand(B,T)
    r_p = R[b_ix, t_ix, s2]
    r = torch.bernoulli(r_p.clamp(0,1)).long()
    return a, r, tran, s
