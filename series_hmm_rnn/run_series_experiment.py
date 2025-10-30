import argparse, torch
from series_hmm_rnn import (
    MFReward, MFChoice, BiasAgent, MBReward,
    TinyMoARNN, SeriesHMMTinyMoARNN,
    two_step_mb_generator, build_Q_seq, make_inputs,
    train_epoch_tiny, eval_epoch_tiny,
    train_epoch_series, eval_epoch_series,
    phase_accuracy_permuted
)

def init_sticky_and_break_symmetry(series, stay=0.97, eps=1e-3):
    with torch.no_grad():
        A = torch.tensor([[stay, 1-stay],[1-stay, stay]], device=series.hmm.log_A.device)
        series.hmm.log_A.copy_(torch.log(A))
        series.by.add_(eps * torch.randn_like(series.by))
        series.Wg.add_(eps * torch.randn_like(series.Wg))

def run(epochs=150, B=64, T=400, hidden_tiny=2, hidden_series=6, K=2, lr=1e-3,
        dwell=120, beta=3.5, seed=1, tau=1.25, sticky=0.97, device=None):
    if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agents = [
        ('Model-free value', MFReward(alpha=0.3, decay=0.0)),
        ('Model-free choice', MFChoice(kappa=0.2, rho=0.0)),
        ('Model-based',  MBReward(p_common=0.7, alpha_state=0.2)),
        ('Bias', BiasAgent(0.0, 0.0)),
    ]

    a_tr, r_tr, t_tr, s_tr = two_step_mb_generator(B=B, T=T, dwell=dwell, beta=beta, seed=seed,   device=device)
    a_te, r_te, t_te, s_te = two_step_mb_generator(B=B, T=T, dwell=dwell, beta=beta, seed=seed+1, device=device)

    tiny = TinyMoARNN(n_agents=len(agents), hidden=hidden_tiny).to(device)
    opt_t = torch.optim.Adam(tiny.parameters(), lr=lr)
    for _ in range(epochs):
        train_epoch_tiny(tiny, opt_t, a_tr, r_tr, t_tr, agents)
    t_loss, t_acc = eval_epoch_tiny(tiny, a_te, r_te, t_te, agents)
    print(f"Tiny  -> NLL: {t_loss:.3f}  Acc: {t_acc:.3f}")

    series = SeriesHMMTinyMoARNN(n_agents=len(agents), hidden=hidden_series, K=K, tau=tau).to(device)
    init_sticky_and_break_symmetry(series, stay=sticky, eps=1e-3)
    opt_s = torch.optim.Adam(series.parameters(), lr=lr)
    for _ in range(epochs):
        train_epoch_series(series, opt_s, a_tr, r_tr, t_tr, agents)
    s_loss, s_acc, gk, lg, _, _ = eval_epoch_series(series, a_te, r_te, t_te, agents)
    gamma = torch.softmax(lg, dim=-1)
    phase_acc_perm, best_perm, _ = phase_accuracy_permuted(gamma, s_te)
    print(f"Series-> NLL: {s_loss:.3f}  Acc: {s_acc:.3f}  PhaseAcc_perm: {phase_acc_perm:.3f}  Perm {best_perm}")
    return {"tiny": tiny, "series": series, "gamma": gamma, "s_te": s_te}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--B', type=int, default=64)
    ap.add_argument('--T', type=int, default=400)
    ap.add_argument('--hidden_tiny', type=int, default=2)
    ap.add_argument('--hidden_series', type=int, default=6)
    ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--dwell', type=int, default=120)
    ap.add_argument('--beta', type=float, default=3.5)
    ap.add_argument('--tau', type=float, default=1.25)
    ap.add_argument('--sticky', type=float, default=0.97)
    args = ap.parse_args()
    run(epochs=args.epochs, B=args.B, T=args.T, hidden_tiny=args.hidden_tiny,
        hidden_series=args.hidden_series, K=args.K, lr=args.lr, seed=args.seed,
        dwell=args.dwell, beta=args.beta, tau=args.tau, sticky=args.sticky)

if __name__ == "__main__":
    import sys
    if 'ipykernel' in sys.argv[0] or 'colab' in sys.argv[0]:
        sys.argv = ['']
    main()
