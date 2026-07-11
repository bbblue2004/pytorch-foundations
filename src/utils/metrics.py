import torch
from tqdm import tqdm


def show_confusion_matrix(metrics):
    print("Confusion matrix")
    print(f"              Pred 0    Pred 1")
    print(f"Actual 0      {metrics['tn']:6d}    {metrics['fp']:6d}")
    print(f"Actual 1      {metrics['fn']:6d}    {metrics['tp']:6d}")
    print()
    print(
        f"Accuracy = {metrics['accuracy']:.4f}, "
        f"Precision = {metrics['precision']:.4f}, "
        f"Recall = {metrics['recall']:.4f}, "
        f"F1-score = {metrics['f1']:.4f}"
    )




@torch.no_grad()      # decorator that disables gradient tracking: pytorch will not build a computational graph inside this function
def linear_regression_metrics(Y: torch.Tensor, Yhat: torch.Tensor) -> dict:
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
def binary_classification_metrics(Y: torch.Tensor, logits: torch.Tensor, threshold=0.5) -> dict:
    logits = logits.view(-1)          # logits : [400, 1] -> [400]
    Y = Y.float().view(-1)            # actually optional, but just in case
    Yhat = torch.sigmoid(logits)
    pred = Yhat >= threshold

    tp = (pred & (Y == 1)).sum().item()     # & for boolean manipulation in pytorch, .item() to get a scalar (int64 generally)
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



@torch.no_grad()
def test_multiclass_accuracy(model, loader, device, nb_classes=10) -> dict:
    model.eval()
    conf_mat = torch.zeros(nb_classes, nb_classes, dtype=torch.int64)

    for images, labels in tqdm(loader, desc="Test dataset"):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        update_conf_mat(conf_mat, labels, preds)

    return metrics_from_conf_mat(conf_mat)


@torch.no_grad()
def update_conf_mat(conf_mat, labels, preds) -> None:
    for label, pred in zip(labels.view(-1), preds.view(-1)):
        conf_mat[label.long(), pred.long()] += 1



@torch.no_grad()
def metrics_from_conf_mat(conf_mat) -> dict:
    nb_classes = len(conf_mat)

    correct = conf_mat.diag().sum().item()
    total = conf_mat.sum().item()
    accuracy = correct / total

    f1s = []
    for c in range(nb_classes):
        tp = conf_mat[c, c].item()
        fp = conf_mat[:, c].sum().item() - tp
        fn = conf_mat[c, :].sum().item() - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        f1s.append(f1)
    
    return {
        "accuracy": accuracy,
        "macro-f1": sum(f1s) / nb_classes,
        "conf_mat": conf_mat
    }