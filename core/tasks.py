# task-specific model fitting (LightGBM, capped windows) and loaders of the stored task results
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import fitting

TASKS = {"D5_h120": 30, "D6_h600": 120}
LEARNED = ["lightgbm", "xgboost", "lstm", "gru"]
NAMES = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "lstm": "LSTM", "gru": "GRU",
         "temperature_rule": "Temperature rule", "rate_rule": "Grid-rise rule", "native_rate_rule": "Native-rate rule"}
FEATURES = ["endpoint", "mean", "std", "maximum", "minimum", "endpoint difference",
            "max one-step rise", "last one-step rise", "availability fraction"]
TREE_CONFIG = dict(depth=3, lr=.03, weight='sqrt')
SEQ_CONFIG = dict(lr=.001, weight='sqrt')


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding='utf-8')


# at most 256 evenly spaced windows per record and class
def cap(ix, keys, y):
    out = []
    for k in np.unique(keys[ix]):
        for cls in [0, 1]:
            ii = ix[(keys[ix] == k) & (y[ix] == cls)]
            if len(ii) > 256:
                ii = ii[np.linspace(0, len(ii) - 1, 256).round().astype(int)]
            out.extend(ii)
    return np.sort(np.array(out, int))


# LightGBM with early stopping on the balanced validation loss; the saved booster is the complete one
# at the lowest validation loss (a boosting round can add no tree, so rounds and trees differ)
def lgb_fit(F, y, keys, tr, va, seed, dest):
    import lightgbm as lgb
    pos = np.sqrt((y[tr] == 0).sum() / max(y[tr].sum(), 1))
    t0 = time.perf_counter()
    best = {'loss': np.inf}

    def snapshot(env):
        value = float(env.evaluation_result_list[0][2])
        if value < best['loss']:
            best.update(loss=value, trees=env.model.num_trees(), text=env.model.model_to_string(num_iteration=-1))
    snapshot.order = 21
    model = lgb.LGBMClassifier(n_estimators=2000, max_depth=3, num_leaves=7, learning_rate=.05, min_child_samples=30,
                               reg_lambda=1, scale_pos_weight=float(pos), subsample=.8, subsample_freq=1, colsample_bytree=.8,
                               random_state=seed, n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
    hist = {}
    model.fit(F[tr], y[tr], eval_set=[(F[va], y[va])], eval_sample_weight=[fitting.balanced_weights(y[va], keys[va])],
              eval_metric='binary_logloss', callbacks=[snapshot, lgb.early_stopping(50, verbose=False), lgb.record_evaluation(hist)])
    lgb.Booster(model_str=best['text']).save_model(str(dest / 'model.txt'), num_iteration=-1)
    loss = hist['valid_0']['binary_logloss']
    pd.DataFrame({'iteration': np.arange(1, len(loss) + 1), 'validation_loss': loss}).to_csv(dest / 'history.csv', index=False)
    info = dict(model='lightgbm', seed=seed, best_epoch=model.best_iteration_, epochs_run=len(loss),
                best_validation_loss=float(min(loss)), fit_s=time.perf_counter() - t0, class_weight=float(pos),
                n_fit_windows=len(tr), n_validation_windows=len(va), best_num_trees=best['trees'])
    dump(dest / 'fit.json', info)
    return info


# risk of a stored task state; d holds X, mask and the window summaries F
def predict(dest, d, ix):
    if (dest / 'model.txt').exists():
        import lightgbm as lgb
        return lgb.Booster(model_file=str(dest / 'model.txt')).predict(d['F'][ix], num_threads=4)
    return fitting.predict(dest, d['X'], d['mask'], d['F'], ix)


# task cohort (results/preprocess/tasks/cohort.csv), indexed by record key
def cohort(tasks_dir):
    return pd.read_csv(Path(tasks_dir) / 'cohort.csv', low_memory=False).set_index('key')


# record-level test decisions of the primary policy (task start, 10 % budget)
def primary_records(comparison_dir):
    x = pd.read_parquet(Path(comparison_dir) / 'per_record.parquet')
    return x[x.policy.eq('task_start') & x.budget.eq(0.1)].copy()


# compact single-channel inputs (results/preprocess/tasks/inputs.npz)
def load_inputs(tasks_dir):
    with np.load(Path(tasks_dir) / 'inputs.npz', allow_pickle=True) as z:
        d = {k: z[k] for k in ['X', 'mask', 'age', 't_end', 'experiment']}
    d['experiment'] = d['experiment'].astype(str)
    d['mask'] = d['mask'].astype(bool)
    return d


def run_dir(comparison_dir, task, fold, model, seed, arm='own'):
    return Path(comparison_dir) / 'runs' / f'{task}__{arm}__f{fold}' / f'{model}_s{seed}'


def stored_risk(comparison_dir, task, fold, model, seed, arm='own'):
    with np.load(run_dir(comparison_dir, task, fold, model, seed, arm) / 'risk.npz') as z:
        return {k: z[k] for k in z.files}
