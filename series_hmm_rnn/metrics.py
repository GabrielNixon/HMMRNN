import torch, itertools

def phase_accuracy_permuted(gamma, s_true):
    """
    gamma: [B,T,K] posteriors; s_true: [B,T] with labels {0..K-1}
    returns: best_acc, best_perm (tuple), confusion[K,K] (pred -> true)
    """
    pred = gamma.argmax(-1)  # [B,T]
    B,T,K = gamma.shape
    C = torch.zeros(K, K, dtype=torch.long, device=gamma.device)
    for i in range(K):
        for j in range(K):
            C[i, j] = ((pred == i) & (s_true == j)).sum()
    total = C.sum().item()
    best_acc, best_perm = 0.0, tuple(range(K))
    for perm in itertools.permutations(range(K)):
        correct = sum(C[i, perm[i]].item() for i in range(K))
        acc = correct / max(1, total)
        if acc > best_acc:
            best_acc, best_perm = acc, perm
    return best_acc, best_perm, C

def align_gamma(gamma, perm):
    return gamma[..., list(perm)]
