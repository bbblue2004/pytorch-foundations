import torch


@torch.no_grad()
def regression_metrics(Y: torch.Tensor, Yhat : torch.Tensor) -> dict:
    residual = Yhat - Y
    sq_err = residual.pow(2)
    mse = sq_err.mean()
    mae = residual.abs().mean()
    rmse = mse.sqrt()
    rss = sq_err.sum()
    tss = (Y - Y.mean()).pow(2).sum()
    r2 = 1 - rss / tss
    return {"mse": mse.item(), "mae": mae.item(), "rmse": rmse.item(), "r2": r2.item()}