import torch, torch.nn as nn
from .utils import make_inputs, build_Q_seq, nll_loss

def train_epoch_tiny(model, opt, actions, rewards, transitions, agents):
    model.train()
    x = make_inputs(actions, rewards, transitions)
    Q_seq = build_Q_seq(actions, rewards, transitions, agents)
    pi_log, _ = model(x, Q_seq)
    loss = nll_loss(pi_log, actions)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        acc = (pi_log.argmax(-1) == actions).float().mean().item()
    return loss.item(), acc

@torch.no_grad()
def eval_epoch_tiny(model, actions, rewards, transitions, agents):
    model.eval()
    x = make_inputs(actions, rewards, transitions)
    Q_seq = build_Q_seq(actions, rewards, transitions, agents)
    pi_log, _ = model(x, Q_seq)
    return nll_loss(pi_log, actions).item(), (pi_log.argmax(-1) == actions).float().mean().item()

def train_epoch_series(model, opt, actions, rewards, transitions, agents):
    model.train()
    x = make_inputs(actions, rewards, transitions)
    Q_seq = build_Q_seq(actions, rewards, transitions, agents)
    pi_log, _, _ = model(x, Q_seq, actions=actions)
    loss = nll_loss(pi_log, actions)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # fixed API
    opt.step()
    with torch.no_grad():
        acc = (pi_log.argmax(-1) == actions).float().mean().item()
    return loss.item(), acc

@torch.no_grad()
def eval_epoch_series(model, actions, rewards, transitions, agents):
    model.eval()
    x = make_inputs(actions, rewards, transitions)
    Q_seq = build_Q_seq(actions, rewards, transitions, agents)
    pi_log, gk, lg = model(x, Q_seq, actions=actions)
    loss = nll_loss(pi_log, actions).item()
    acc  = (pi_log.argmax(-1) == actions).float().mean().item()
    return loss, acc, gk, lg
