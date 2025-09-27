import torch, yaml, argparse
from series_hmm_rnn.data import synthetic_batch
from series_hmm_rnn.build import assemble_agents, build_Q_seq, make_inputs
from series_hmm_rnn.models import TinyMoARNN
from series_hmm_rnn.train import train_epoch, eval_epoch

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/tiny.yaml")
args = p.parse_args()
cfg = yaml.safe_load(open(args.config))
device = "cuda" if torch.cuda.is_available() else "cpu"

agents = assemble_agents(cfg["agents"])
model = TinyMoARNN(n_agents=len(agents), hidden=cfg["model"]["hidden"]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=cfg["optim"]["lr"])

a_tr, r_tr, t_tr = synthetic_batch(B=cfg["data"]["batch"], T=cfg["data"]["T"], seed=1, device=device)
a_te, r_te, t_te = synthetic_batch(B=cfg["data"]["batch"], T=cfg["data"]["T"], seed=2, device=device)

for _ in range(cfg["train"]["epochs"]):
    tr = train_epoch(model, opt, a_tr, r_tr, t_tr, agents)
    ev = eval_epoch(model, a_te, r_te, t_te, agents)

torch.save(model.state_dict(), "tiny_moa_rnn.pt")
