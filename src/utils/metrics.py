import torch


@torch.no_grad()      # decorator that disables gradient tracking: pytorch will not build a computational graph inside this function
def linear_regression_metrics(Y: torch.Tensor, Yhat : torch.Tensor) -> dict:
    residual = Yhat - Y
    sq_err = residual.pow(2)    # or **2, but it's used more often apparently
    mse = sq_err.mean()
    mae = residual.abs().mean()
    rmse = mse.sqrt()
    rss = sq_err.sum()
    tss = (Y - Y.mean()).pow(2).sum()
    r2 = 1 - rss / tss
    return {"mse": mse.item(), "mae": mae.item(), "rmse": rmse.item(), "r2": r2.item()}


@torch.no_grad()
def logistic_regression_metrics(Y: torch.Tensor, logits : torch.Tensor, threshold=0.5) -> dict:
    logits = logits.view(-1)          # logits : [400, 1] -> [400]
    Y = Y.float().view(-1)            # actually optional, but just in case
    Yhat = torch.sigmoid(logits)
    pred = Yhat >= threshold

    tp = (pred & (Y == 1)).sum().item()     # & for boolean manipulation in pytorch, .item() to get a float
    tn = (~pred & (Y == 0)).sum().item()    # ~pred instead of !pred
    fp = (pred & (Y == 0)).sum().item()
    fn = (~pred & (Y == 1)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }