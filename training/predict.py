#!/usr/bin/env python3
"""
ANZSRC FoR 2020 classifier: groups for new publications.

    python predict.py --title "Paper title" --abstract "Abstract ..."
    python predict.py --input new_papers.xlsx --output predictions.xlsx
    python predict.py --input new_papers.csv --title-col Title --abstract-col Abstract --output predictions.csv
    python predict.py --input new_papers.xlsx --output top3.xlsx --threshold 0 --max-labels 3   # always three groups

Uses the models and a combination saved by step 4 (models/<ensemble>/config.json and the models it names). By default
the newest one that exists: ensemble_final, then ensemble_large, ensemble_clean, ensemble; --ensemble picks another.
Run it from the training folder or keep models/ and data/labels.json next to it. Runs on the GPU when there is
one, otherwise on the CPU (slower, but fine for a few hundred records).

Input: CSV or Excel with a title and an abstract column (default names Title and Abstract, any case).
Output: the input columns plus
  pred_codes      groups the model assigns, e.g. 3103;3104 (at least one per record)
  pred_groups     the same with names
  divisions       their divisions
  top5            the five most probable groups with probabilities
  max_probability probability of the most probable group (low values = the model is unsure)
--max-labels caps the number of groups per record; --threshold overrides the threshold chosen in step 4
(higher = fewer, more precise groups).
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import MODELS, load_labels, say, tfidf_text   # noqa: E402


PREFERRED = ('ensemble_final', 'ensemble_large', 'ensemble_clean', 'ensemble')   # newest first

def parse_args():
    p = argparse.ArgumentParser(description='ANZSRC FoR 2020 groups for new publications')
    p.add_argument('--title', default='')
    p.add_argument('--abstract', default='')
    p.add_argument('--input')
    p.add_argument('--output')
    p.add_argument('--title-col', default='Title')
    p.add_argument('--abstract-col', default='Abstract')
    p.add_argument('--threshold', type=float, default=None)
    p.add_argument('--max-labels', type=int, default=None)
    p.add_argument('--ensemble', default=None,
                   help='which combination from step 4 (models/<ensemble>); default: the newest that exists of '
                        + ', '.join(PREFERRED))
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--device', default=None, help='cuda or cpu (default: cuda when available)')
    return p.parse_args()


def find_col(df, name):
    for c in df.columns:
        if str(c).strip().lower() == name.strip().lower():
            return c
    raise SystemExit(f'У файлі немає колонки «{name}». Є: {", ".join(map(str, df.columns))}')


def read_input(args):
    if not args.input:
        if not (args.title or args.abstract):
            raise SystemExit('Дай --title і --abstract або --input файл.')
        return pd.DataFrame({'Title': [args.title], 'Abstract': [args.abstract]}), 'Title', 'Abstract'
    path = Path(args.input)
    if path.suffix.lower() in ('.xlsx', '.xlsm', '.xls'):
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    df = df.fillna('')
    return df, find_col(df, args.title_col), find_col(df, args.abstract_col)


def baseline_probs(d, df):
    import joblib
    vec = joblib.load(d / 'vectorizer.joblib')
    W = np.load(d / 'coef.npy').astype(np.float32)
    b = np.load(d / 'intercept.npy').astype(np.float32)
    X = vec.transform(tfidf_text(df))
    return 1.0 / (1.0 + np.exp(-(X @ W.T + b)))


def transformer_probs(d, df, dev, batch):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    cfg = json.loads((d / 'best' / 'config.json').read_text(encoding='utf-8'))
    tok = AutoTokenizer.from_pretrained(d / 'best' / 'encoder')
    enc = AutoModel.from_pretrained(d / 'best' / 'encoder')
    head = torch.nn.Linear(enc.config.hidden_size, len(cfg['classes']))
    head.load_state_dict(torch.load(d / 'best' / 'head.pt', map_location='cpu'))
    enc.to(dev).eval()
    head.to(dev).eval()
    # bf16 only where the GPU has it (Ampere and newer); is_bf16_supported() also says yes where it is only emulated
    amp = torch.bfloat16 if dev.type == 'cpu' or torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16
    texts = (df.title + tok.sep_token + df.abstract).tolist()
    ids = tok(texts, truncation=True, max_length=cfg['max_len'])['input_ids']
    order = np.argsort([len(x) for x in ids], kind='stable')
    out = np.zeros((len(ids), len(cfg['classes'])), dtype=np.float32)
    with torch.inference_mode():
        for k in range(0, len(order), batch):
            b = order[k:k + batch]
            L = max(len(ids[i]) for i in b)
            x = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
            m = torch.zeros((len(b), L), dtype=torch.long)
            for r, i in enumerate(b):
                x[r, :len(ids[i])] = torch.tensor(ids[i])
                m[r, :len(ids[i])] = 1
            with torch.autocast(dev.type, dtype=amp):
                h = enc(input_ids=x.to(dev), attention_mask=m.to(dev)).last_hidden_state[:, 0]
                logits = head(h)
            out[b] = torch.sigmoid(logits.float()).cpu().numpy()
    return out, cfg['classes']


def logit(p):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def main():
    args = parse_args()
    if args.ensemble is None:
        args.ensemble = next((e for e in PREFERRED if (MODELS / e / 'config.json').exists()), 'ensemble')
    say(f'Поєднання: models/{args.ensemble}')
    cfg = json.loads((MODELS / args.ensemble / 'config.json').read_text(encoding='utf-8'))
    classes = cfg['classes']
    try:
        _, names, divn = load_labels()
    except FileNotFoundError:
        names, divn = {}, {}
    df, tcol, acol = read_input(args)
    work = pd.DataFrame({'title': df[tcol].astype(str).str.strip(), 'abstract': df[acol].astype(str).str.strip()})

    used = [(n, w) for n, w in cfg['weights'].items() if w]
    dev = None
    if any(n != 'baseline' for n, _ in used):
        import torch
        dev = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    z = 0.0
    for name, w in used:
        d = MODELS / name
        if (d / 'vectorizer.joblib').exists():
            P = baseline_probs(d, work)
        else:
            P, cls = transformer_probs(d, work, dev, args.batch)
            if cls != classes:
                raise SystemExit(f'У моделі {name} інший порядок груп, ніж у поєднанні: перезапусти крок 4.')
        z = z + w * (logit(P) - logit(np.array(cfg['model_thresholds'][name])))
    P = 1.0 / (1.0 + np.exp(-z))

    thr = cfg['threshold'] if args.threshold is None else args.threshold
    order = np.argsort(-P, 1)
    rows = []
    for i in range(len(P)):
        chosen = [j for j in order[i] if P[i, j] >= thr] or [order[i, 0]]
        if args.max_labels:
            chosen = chosen[:args.max_labels]
        codes = [classes[j] for j in chosen]
        rows.append({'pred_codes': ';'.join(codes),
                     'pred_groups': '; '.join(f'{c} {names.get(c, "")}'.strip() for c in codes),
                     'divisions': '; '.join(dict.fromkeys(f'{c[:2]} {divn.get(c[:2], "").capitalize()}'.strip() for c in codes)),
                     'top5': '; '.join(f'{classes[j]} {names.get(classes[j], "")} ({P[i, j]:.2f})' for j in order[i, :5]),
                     'max_probability': round(float(P[i, order[i, 0]]), 3)})
    res = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    if not args.input:
        r = rows[0]
        say(f'Групи: {r["pred_groups"]}\nРозділи: {r["divisions"]}\nТоп-5: {r["top5"]}')
        return
    out = Path(args.output or Path(args.input).with_name(Path(args.input).stem + '_predictions.xlsx'))
    if out.suffix.lower() in ('.xlsx', '.xlsm'):
        res.to_excel(out, index=False)
    else:
        res.to_csv(out, index=False, encoding='utf-8-sig')
    unsure = (res.max_probability < 0.5).sum()
    say(f'Готово: {len(res)} записів → {out}' + (f'\nМодель не впевнена (max_probability < 0.5) у {unsure} записах.' if unsure else ''))


if __name__ == '__main__':
    main()
