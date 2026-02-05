import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


def linear_probe(model, train_loader, val_loader, num_classes: int, device, epochs: int = 5, log_event=None):
    model.eval()
    feat_dim = model.backbone.num_features
    clf = nn.Linear(feat_dim, num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-6)
    for epoch in range(epochs):
        clf.train()
        total = 0.0
        prog = tqdm(train_loader, desc=f"probe epoch {epoch+1}/{epochs}", leave=False)
        for views, y in prog:
            views = views.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
                # ensure target is 1D class indices for cross_entropy
            if y.dim() > 1:
                y = y.view(-1)
            
            with torch.no_grad():
                emb, _ = model(views)
                feats = emb[:, 0, :]
            logits = clf(feats)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            prog.set_postfix(loss=loss.item())
        val_acc = evaluate_probe(model, clf, val_loader, device)
        if log_event:
            log_event(
                {
                    "phase": "probe",
                    "epoch": epoch + 1,
                    "train_loss": total / len(train_loader),
                    "val_acc": val_acc,
                }
            )
        print(f"[probe] epoch {epoch+1}/{epochs} loss={total/len(train_loader):.4f} val_acc={val_acc:.4f}")
    return clf


def evaluate_probe(model, clf, loader, device):
    model.eval()
    clf.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for views, y in loader:
            views = views.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if y.dim() > 1:
                y = y.view(-1)
            emb, _ = model(views)
            logits = clf(emb[:, 0, :])
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(1, total)


def collect_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for views, y in loader:
            views = views.to(device, non_blocking=True)
            emb, _ = model(views)
            feats.append(emb[:, 0, :].cpu())
            # ensure labels are 1D on CPU
            y = y.view(-1).cpu()
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def radius_knn(train_feat, train_label, test_feat, num_classes: int, radius: float, k: int):
    preds = []
    for x in test_feat:
        dists = torch.cdist(x.unsqueeze(0), train_feat, p=2).squeeze(0)
        within = dists <= radius
        if within.any():
            neighbors = train_label[within]
            weights = torch.ones_like(neighbors, dtype=torch.float)
        else:
            vals, idx = torch.topk(dists, k, largest=False)
            neighbors = train_label[idx]
            weights = 1.0 / (vals + 1e-6)
        votes = torch.zeros(num_classes, dtype=torch.float)
        votes.scatter_add_(0, neighbors, weights)
        preds.append(votes.argmax().item())
    return torch.tensor(preds)


def knn_eval(model, train_loader, val_loader, num_classes: int, radius: float, k: int, device, log_event=None):
    train_feat, train_label = collect_features(model, train_loader, device)
    val_feat, val_label = collect_features(model, val_loader, device)
    preds = radius_knn(train_feat, train_label, val_feat, num_classes=num_classes, radius=radius, k=k)
    acc = (preds == val_label).float().mean().item()
    if log_event:
        log_event({"phase": "knn", "radius": radius, "k": k, "acc": acc})
    print(f"[knn] radius={radius} k={k} acc={acc:.4f}")
    return acc
