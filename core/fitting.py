# validation-selected fitting of XGBoost and sequence models, and replay of stored states
import gc
import hashlib
import json
import random
import time

import numpy as np
import pandas as pd
import torch

from core import models

SEQ_CONFIGS = [dict(lr=.001, weight='ratio'), dict(lr=.0003, weight='ratio'), dict(lr=.001, weight='sqrt')]
TREE_CONFIGS = [dict(depth=5, lr=.08, weight='ratio'), dict(depth=3, lr=.03, weight='ratio'), dict(depth=3, lr=.03, weight='sqrt')]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def configuration(model, i):
    return (TREE_CONFIGS if model == 'xgboost' else SEQ_CONFIGS)[i]


# validation weights: each record and class gets equal total weight
def balanced_weights(y, keys):
    w = np.zeros(len(y), np.float64)
    for cls in [0, 1]:
        ks = np.unique(keys[y == cls])
        if not len(ks):
            raise ValueError('Validation lacks target class')
        for k in ks:
            ix = (keys == k) & (y == cls)
            w[ix] = .5 / len(ks) / ix.sum()
    assert np.isclose(w.sum(), 1)
    return w


# sequence model with warm-up, plateau schedule and early stopping on the validation loss
def seq_fit(model, X, M, y, keys, tr, va, seed, cfg, dest, max_epochs=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    dev = models.device()
    t0 = time.perf_counter()
    vals = np.where(M[tr], X[tr], np.nan)
    with np.errstate(invalid='ignore'):
        mu = np.nan_to_num(np.nanmean(vals, axis=(0, 1)))
        sd = np.nan_to_num(np.nanstd(vals, axis=(0, 1)), nan=1.)
    sd = np.where(sd > 1e-6, sd, 1.).astype('float32')
    mu = mu.astype('float32')
    del vals
    xx = torch.as_tensor(models.prep(X[tr], M[tr], mu, sd), device=dev)
    yy = torch.as_tensor(y[tr], device=dev)
    vx = torch.as_tensor(models.prep(X[va], M[va], mu, sd), device=dev)
    vy = torch.as_tensor(y[va], device=dev)
    vw = torch.as_tensor(balanced_weights(y[va], keys[va]), device=dev, dtype=torch.float32)
    net = models.build(model, X.shape[2] * 2, X.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'] / 3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=.5, patience=3, threshold=1e-4,
                                                       threshold_mode='abs', min_lr=1e-6)
    ratio = float((y[tr] == 0).sum()) / max(float(y[tr].sum()), 1.)
    pos = ratio if cfg['weight'] == 'ratio' else np.sqrt(ratio)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos, device=dev))
    best = np.inf
    significant = np.inf
    stale = 0
    best_ep = 0
    best_state = None
    state6 = None
    history = []
    for ep in range(1, max_epochs + 1):
        if ep <= 3:
            for g in opt.param_groups:
                g['lr'] = cfg['lr'] * ep / 3
        net.train()
        perm = torch.randperm(len(tr), device=dev)
        tl = 0.
        for start in range(0, len(tr), 1024):
            ii = perm[start:start + 1024]
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(xx[ii]), yy[ii])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
            tl += float(loss.detach()) * len(ii)
        net.eval()
        vl = 0.
        with torch.no_grad():
            for start in range(0, len(va), 1024):
                logits = net(vx[start:start + 1024])
                ll = torch.nn.functional.binary_cross_entropy_with_logits(logits, vy[start:start + 1024], reduction='none')
                vl += float((ll * vw[start:start + 1024]).sum())
        if not np.isfinite(vl):
            raise ValueError('Non-finite validation loss')
        history.append(dict(epoch=ep, train_loss=tl / len(tr), validation_loss=vl, lr=opt.param_groups[0]['lr']))
        if vl < best:
            best = vl
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if ep == 6:
            state6 = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if vl < significant - 1e-4:
            significant = vl
            stale = 0
        else:
            stale += 1
        if ep >= 3:
            sched.step(vl)
        if ep >= 10 and stale >= 10:
            break
    info = dict(model=model, seed=seed, config=cfg, epochs_run=ep, best_epoch=best_ep, best_validation_loss=best,
                class_weight=float(pos), fit_s=time.perf_counter() - t0, n_fit_windows=len(tr), n_validation_windows=len(va))
    checkpoint = dict(state_dict=best_state, mu=mu, sd=sd, model=model, config=cfg, seed=seed, window=X.shape[1], c_in=X.shape[2] * 2)
    torch.save(checkpoint, dest / 'model.pt')
    if state6 is not None:
        torch.save(dict(checkpoint, state_dict=state6), dest / 'epoch6.pt')
    pd.DataFrame(history).to_csv(dest / 'history.csv', index=False)
    write_json(dest / 'fit.json', info)
    del net, xx, yy, vx, vy, vw, opt, sched
    gc.collect()
    torch.cuda.empty_cache()
    return info


# XGBoost with early stopping on the balanced validation loss
def tree_fit(F, y, keys, tr, va, seed, cfg, dest):
    import xgboost as xgb
    ratio = float((y[tr] == 0).sum()) / max(float(y[tr].sum()), 1.)
    pos = ratio if cfg['weight'] == 'ratio' else np.sqrt(ratio)
    model = xgb.XGBClassifier(n_estimators=2000, max_depth=cfg['depth'], learning_rate=cfg['lr'], subsample=.8,
                              colsample_bytree=.8, min_child_weight=2, scale_pos_weight=pos, tree_method='hist', device='cpu',
                              eval_metric='logloss', early_stopping_rounds=50, random_state=seed, n_jobs=4, verbosity=0)
    t0 = time.perf_counter()
    model.fit(F[tr], y[tr], eval_set=[(F[va], y[va])], sample_weight_eval_set=[balanced_weights(y[va], keys[va])], verbose=False)
    model.save_model(dest / 'model.ubj')
    hist = model.evals_result()['validation_0']['logloss']
    info = dict(model='xgboost', seed=seed, config=cfg, epochs_run=len(hist), best_epoch=int(model.best_iteration) + 1,
                best_validation_loss=float(model.best_score), class_weight=float(pos), fit_s=time.perf_counter() - t0,
                n_fit_windows=len(tr), n_validation_windows=len(va))
    pd.DataFrame({'iteration': np.arange(1, len(hist) + 1), 'validation_loss': hist}).to_csv(dest / 'history.csv', index=False)
    write_json(dest / 'fit.json', info)
    return info


# risk of a stored XGBoost or sequence-model state on selected windows
def predict(folder, X, M, F, ix, checkpoint='model.pt'):
    if (folder / 'model.ubj').exists():
        import xgboost as xgb
        m = xgb.XGBClassifier()
        m.load_model(bytearray((folder / 'model.ubj').read_bytes()))
        m.set_params(device='cpu', n_jobs=4)
        return m.predict_proba(F[ix])[:, 1]
    cp = torch.load(folder / checkpoint, map_location='cpu', weights_only=False)
    dev = models.device()
    net = models.build(cp['model'], cp['c_in'], cp['window']).to(dev)
    net.load_state_dict(cp['state_dict'])
    net.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(ix), 1024):
            ii = ix[start:start + 1024]
            f = models.prep(X[ii], M[ii], cp['mu'], cp['sd'])
            out.append(torch.sigmoid(net(torch.as_tensor(f, device=dev))).cpu().numpy())
    del net
    return np.concatenate(out)
