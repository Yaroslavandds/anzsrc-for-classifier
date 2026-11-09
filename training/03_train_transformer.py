#!/usr/bin/env python3
"""
ANZSRC FoR 2020 classifier, step 3: fine-tune a BERT-type encoder for the 213 groups.

    python 03_train_transformer.py --smoke     # 3-5 minutes on a small sample: checks that everything works
    python 03_train_transformer.py             # the real run: about 2-3 hours on an RTX 3050 Ti Laptop (4 GB)
    python 03_train_transformer.py --model allenai/scibert_scivocab_uncased     # SciBERT instead of SPECTER2
    python 03_train_transformer.py --train train_clean                          # on the cleaned train (step 6)
    python 03_train_transformer.py --model FacebookAI/roberta-base --train train_clean    # RoBERTa, cleaned train
    python 03_train_transformer.py --model answerdotai/ModernBERT-large --train train_clean --lora 32 --batch 16
                                                       # a large model (395M) with LoRA: fits into 4 GB, 7-8 hours

Needs data/ from step 1, common.py, PyTorch with CUDA and transformers (step 0).
Each model goes to its own folder, models/<name> (specter2, scibert, ...; with --train train_clean:
specter2_clean, ...; --name overrides), so step 4 can combine any of them.

Model: the encoder (default SPECTER2, allenai/specter2_base) + one linear layer over the [CLS] vector,
213 outputs (one per group).
Input: title [SEP] abstract, at most --max-len tokens.
Training:
* loss: binary cross-entropy per group, positives weighted by sqrt(negatives / positives), capped at 30,
  so that rare groups count; the output biases start at each group's prior;
* AdamW, learning rate 3e-5 for the encoder and 3e-4 for the output layer, 6 % warm-up then linear decay,
  weight decay 0.01, gradient clipping at 1.0;
* --lora R: instead of all encoder weights, only low-rank additions (rank R) to every linear layer are trained
  (LoRA), the encoder itself stays frozen in bf16. A large model then needs ~2.5 GB of GPU memory instead of
  ~7 GB; learning rate 2e-4 for the additions. At the end they are merged into the weights, so the saved model
  is an ordinary one (steps 4-5 and predict.py use it as any other);
* bf16 mixed precision, gradient checkpointing (less GPU memory, ~30 % slower), batches of similar length;
* after each epoch: probabilities on val, one shared threshold tuned for micro-F1, metrics; the epoch with
  the best (micro-F1 + macro-F1) / 2 is kept.
The state is saved every --save-every-min minutes and after each epoch (models/<name>/last.pt). If the run
stops (Ctrl+C, power, crash), run the same command again: it continues from the last save. --fresh starts over.
While it runs the script asks Windows not to go to sleep (closing the lid may still send it to sleep).

Output: models/<name>/best/ (the model), models/<name>/val_probs.npy and test_probs.npy (for steps 4-5),
        reports/03_<name>.json. Test probabilities are saved, not scored.
"""
import argparse
import gc
import json
import os
import sys
import time

os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import torch
from tqdm import tqdm

from common import (MODELS, REPORTS, evaluate, evaluate_by_origin, label_matrix, load_labels, load_split,
                    save_json, say, show, tune_threshold)


def parse_args():
    p = argparse.ArgumentParser(description='Fine-tune SPECTER2 for ANZSRC FoR 2020 groups')
    p.add_argument('--model', default='allenai/specter2_base')
    p.add_argument('--name', default=None, help='output folder models/<name> (default: from --model, e.g. specter2, scibert)')
    p.add_argument('--train', default='train', help='training part in data/ (train or train_clean)')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=None, help='encoder learning rate (default 3e-5; with --lora 2e-4)')
    p.add_argument('--head-lr', type=float, default=3e-4)
    p.add_argument('--lora', type=int, default=0, help='train LoRA additions of this rank instead of the whole encoder')
    p.add_argument('--lora-alpha', type=float, default=None, help='LoRA scale numerator (default 2 x rank)')
    p.add_argument('--lora-dropout', type=float, default=0.05)
    p.add_argument('--max-len', type=int, default=512)
    p.add_argument('--no-checkpointing', action='store_true', help='faster, needs more GPU memory (use --batch 8)')
    p.add_argument('--save-every-min', type=float, default=20.0)
    p.add_argument('--no-state', action='store_true', help='never write last.pt (no resuming; e.g. in a cloud notebook)')
    p.add_argument('--time-budget-h', type=float, default=None,
                   help='finish within this many hours: training stops early and the best epoch so far is kept')
    p.add_argument('--reserve-min', type=float, default=45.0, help='time kept for the final steps under --time-budget-h')
    p.add_argument('--fresh', action='store_true', help='ignore a saved state and start over')
    p.add_argument('--smoke', action='store_true', help='quick check on a small sample (separate output folder)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    args.lr = args.lr or (2e-4 if args.lora else 3e-5)
    args.lora_alpha = args.lora_alpha or 2 * args.lora
    return args


def model_name(model):
    """short folder name: allenai/specter2_base -> specter2, answerdotai/ModernBERT-large -> modernbert_large"""
    base = str(model).replace('\\', '/').rstrip('/').split('/')[-1].lower()
    for key in ('modernbert', 'specter2', 'scibert', 'pubmedbert', 'biomedbert', 'roberta', 'deberta'):
        if key in base:
            return key + ('_large' if 'large' in base else '')
    return ''.join(ch if ch.isalnum() else '_' for ch in base)


class LoRALinear(torch.nn.Module):
    """a frozen linear layer plus a trainable low-rank addition: W x + (alpha / r) B A x (B starts at zero)"""
    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.lora_A = torch.nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, r))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        F = torch.nn.functional
        return self.base(x) + F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B) * self.scale

    def merged(self):
        """an ordinary fp32 linear layer with the addition folded into the weights"""
        lin = torch.nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            lin.weight.copy_(self.base.weight.float() + self.scale * (self.lora_B.float() @ self.lora_A.float()))
            if self.base.bias is not None:
                lin.bias.copy_(self.base.bias.float())
        return lin


def _replace(root, name, module):
    parent = root.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else root
    setattr(parent, name.rsplit('.', 1)[-1], module)


def add_lora(encoder, r, alpha, dropout):
    """freeze the encoder, wrap every linear layer; returns the number of trainable parameters"""
    for p in encoder.parameters():
        p.requires_grad_(False)
    for name, m in [(n, m) for n, m in encoder.named_modules() if isinstance(m, torch.nn.Linear)]:
        _replace(encoder, name, LoRALinear(m, r, alpha, dropout))
    return sum(p.numel() for p in encoder.parameters() if p.requires_grad)


def merge_lora(encoder):
    for name, m in [(n, m) for n, m in encoder.named_modules() if isinstance(m, LoRALinear)]:
        _replace(encoder, name, m.merged())
    return encoder.float()


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, sd):
    params = dict(model.named_parameters())
    want = {n for n, p in params.items() if p.requires_grad}
    if set(sd) != want:
        sys.exit('Збережений стан не підходить до цієї моделі. Запусти з тими самими параметрами або додай --fresh.')
    with torch.no_grad():
        for n, v in sd.items():
            params[n].copy_(v)


def keep_awake(on):
    """ask Windows not to sleep while training (released when the script ends)"""
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


class Classifier(torch.nn.Module):
    def __init__(self, encoder, n_labels):
        super().__init__()
        self.encoder = encoder
        self.dropout = torch.nn.Dropout(0.1)
        self.head = torch.nn.Linear(encoder.config.hidden_size, n_labels)

    def forward(self, ids, mask):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(self.dropout(h))


def tokenize(tok, df, max_len):
    texts = (df.title + tok.sep_token + df.abstract).tolist()
    ids = tok(texts, truncation=True, max_length=max_len)['input_ids']
    return [np.asarray(x, dtype=np.int64) for x in ids]


def length_batches(lengths, batch, rng, mega=50):
    """shuffle, sort by length inside chunks of `mega` batches (less padding), shuffle the batches"""
    idx = rng.permutation(len(lengths))
    out = []
    for k in range(0, len(idx), batch * mega):
        chunk = idx[k:k + batch * mega]
        chunk = chunk[np.argsort(lengths[chunk], kind='stable')]
        out += [chunk[i:i + batch] for i in range(0, len(chunk), batch)]
    return [out[i] for i in rng.permutation(len(out))]


def collate(seqs, pad_id, dev):
    L = max(len(s) for s in seqs)
    ids = np.full((len(seqs), L), pad_id, dtype=np.int64)
    mask = np.zeros((len(seqs), L), dtype=np.int64)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
        mask[i, :len(s)] = 1
    return torch.from_numpy(ids).to(dev, non_blocking=True), torch.from_numpy(mask).to(dev, non_blocking=True)


@torch.no_grad()
def predict(model, seqs, pad_id, dev, amp, batch=64):
    was_training = model.training
    model.eval()
    order = np.argsort([len(s) for s in seqs], kind='stable')
    out = np.zeros((len(seqs), model.head.out_features), dtype=np.float32)
    for k in range(0, len(order), batch):
        b = order[k:k + batch]
        ids, mask = collate([seqs[i] for i in b], pad_id, dev)
        with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
            logits = model(ids, mask)
        out[b] = torch.sigmoid(logits.float()).cpu().numpy()
    model.train(was_training)
    return out


def save_best(model, tok, out_dir, cfg, lora=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    if lora:                                      # only the additions and the output layer; merged at the end
        torch.save(trainable_state(model), out_dir / 'lora.pt')
    else:
        model.encoder.save_pretrained(out_dir / 'encoder')
        tok.save_pretrained(out_dir / 'encoder')
        torch.save(model.head.state_dict(), out_dir / 'head.pt')
    save_json(out_dir / 'config.json', cfg)


def save_state(path, **state):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        sys.exit('Відеокарта недоступна для PyTorch: перевір крок 0.')
    dev = torch.device(args.device)
    # bf16 only on GPUs that have it (Ampere and newer); torch.cuda.is_bf16_supported() also says yes on a T4,
    # where bf16 is only emulated and very slow, so the compute capability is checked instead
    amp = torch.bfloat16 if dev.type == 'cpu' or torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16
    deadline = t0 + 3600 * args.time_budget_h - 60 * args.reserve_min if args.time_budget_h else None
    torch.set_float32_matmul_precision('high')
    name = args.name or model_name(args.model) + ('' if args.train == 'train' else '_' + args.train.replace('train_', ''))
    out = MODELS / (f'{name}_smoke' if args.smoke else name)
    report = REPORTS / (f'03_{name}_smoke.json' if args.smoke else f'03_{name}.json')
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / 'last.pt'

    say(f'=== Крок 3: дотренування {args.model} на {args.train} -> models/{out.name}' + (' (швидка перевірка)' if args.smoke else '') + ' ===\n')
    classes, names, _ = load_labels()
    tr, va, te = load_split(args.train), load_split('val'), load_split('test')
    n_train_full = len(tr)
    if args.smoke:
        tr = tr.sample(1500, random_state=0).reset_index(drop=True)
        va = va.sample(500, random_state=0).reset_index(drop=True)
        te = te.sample(500, random_state=0).reset_index(drop=True)
        args.epochs, args.save_every_min = 1, 1.0
    Ytr, Yva = label_matrix(tr.labels, classes), label_matrix(va.labels, classes)

    say('Токенізую тексти...')
    tok = AutoTokenizer.from_pretrained(args.model)
    seq_tr, seq_va, seq_te = (tokenize(tok, d, args.max_len) for d in (tr, va, te))
    len_tr = np.array([len(s) for s in seq_tr])
    say(f'  train {len(seq_tr)}, val {len(seq_va)}, test {len(seq_te)}; середня довжина {len_tr.mean():.0f} токенів, '
        f'обрізано до {args.max_len}: {np.mean(len_tr >= args.max_len):.1%}')

    # with LoRA the frozen encoder is kept in bf16 (half the memory); otherwise all weights are trained in fp32
    load_dtype = amp if args.lora else torch.float32
    encoder = AutoModel.from_pretrained(args.model, dtype=load_dtype)
    n_all = sum(p.numel() for p in encoder.parameters())
    if args.lora:
        n_train = add_lora(encoder, args.lora, args.lora_alpha, args.lora_dropout)
        say(f'LoRA, ранг {args.lora}: навчаються {n_train / 1e6:.1f} млн параметрів із {n_all / 1e6:.0f} млн')
    if not args.no_checkpointing:
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = Classifier(encoder, len(classes))
    eval_batch = 32 if encoder.config.hidden_size >= 1024 else 64     # large models: smaller batches on val/test
    pos = Ytr.sum(0).astype(np.float64)
    pos_weight = np.clip(np.sqrt((len(Ytr) - pos) / np.maximum(pos, 1)), 1.0, 30.0)
    prior = np.clip(pos / len(Ytr), 1e-6, 1 - 1e-6)
    with torch.no_grad():
        model.head.bias.copy_(torch.tensor(np.log(pos_weight * prior / (1 - prior)), dtype=torch.float32))
    model.to(dev).train()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=dev))

    no_decay = lambda n, p: p.ndim < 2 or 'LayerNorm' in n or n.endswith('.bias')
    enc_params = [(n, p) for n, p in model.encoder.named_parameters() if p.requires_grad]
    train_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {'params': [p for n, p in enc_params if not no_decay(n, p)], 'lr': args.lr, 'weight_decay': 0.01},
        {'params': [p for n, p in enc_params if no_decay(n, p)], 'lr': args.lr, 'weight_decay': 0.0},
        {'params': list(model.head.parameters()), 'lr': args.head_lr, 'weight_decay': 0.0}])
    steps_per_epoch = (len(seq_tr) + args.batch - 1) // args.batch
    total = steps_per_epoch * args.epochs
    warm = max(1, int(0.06 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    scaler = torch.amp.GradScaler('cuda', enabled=(dev.type == 'cuda' and amp == torch.float16))

    start_epoch, start_step, best, history = 0, 0, -1.0, []
    keys = ('model', 'train', 'epochs', 'batch', 'lr', 'head_lr', 'max_len', 'no_checkpointing', 'seed',
            'lora', 'lora_alpha')
    if state_path.exists() and not args.fresh:
        st = torch.load(state_path, map_location='cpu', weights_only=False)
        saved = {'lora': 0, 'lora_alpha': 0, **st['args']}         # states saved before LoRA existed
        changed = [k for k in keys if saved.get(k) != vars(args).get(k)]
        if changed:
            sys.exit(f'Збережений стан має інші параметри ({", ".join(changed)}). Запусти з тими самими параметрами '
                     f'або додай --fresh, щоб почати заново.')
        if args.lora:
            load_trainable(model, st['model'])
        else:
            model.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
        if st.get('scaler'):
            scaler.load_state_dict(st['scaler'])
        start_epoch, start_step, best, history = st['epoch'], st['step'], st['best'], st['history']
        say(f'Продовжую з епохи {start_epoch + 1}, крок {start_step} зі {steps_per_epoch}')

    def checkpoint(epoch, step):
        if args.no_state:
            return
        save_state(state_path, model=trainable_state(model) if args.lora else model.state_dict(),
                   opt=opt.state_dict(), sched=sched.state_dict(),
                   scaler=scaler.state_dict() if scaler.is_enabled() else None, epoch=epoch, step=step,
                   best=best, history=history, args=vars(args))

    pad = tok.pad_token_id
    keep_awake(True)
    say(f'\nНавчання: {args.epochs} епохи по {steps_per_epoch} кроків, батч {args.batch}, '
        f'{"bf16" if amp == torch.bfloat16 else "fp16"}, gradient checkpointing: {"ні" if args.no_checkpointing else "так"}')
    if not args.no_state:
        say('Зупинити можна Ctrl+C: стан збережеться, а та сама команда продовжить навчання.')
    if deadline:
        say(f'Обмеження часу: {args.time_budget_h} год від запуску (з них {args.reserve_min:.0f} хв на завершення).')
    say('')
    epoch, step = start_epoch, start_step
    step_times, bar, skipped, out_of_time = [], None, 0, False
    try:
        for epoch in range(start_epoch, args.epochs):
            batches = length_batches(len_tr, args.batch, np.random.RandomState(args.seed + epoch))
            first = start_step if epoch == start_epoch else 0
            bar = tqdm(total=len(batches), initial=first, desc=f'Епоха {epoch + 1}/{args.epochs}', unit='крок',
                       dynamic_ncols=True)
            last_save, ema = time.time(), None
            for step in range(first, len(batches)):
                if deadline and time.time() > deadline:
                    out_of_time = True
                    break
                ts = time.time()
                b = batches[step]
                ids, mask = collate([seq_tr[i] for i in b], pad, dev)
                y = torch.from_numpy(Ytr[b].astype(np.float32)).to(dev, non_blocking=True)
                with torch.autocast(dev.type, dtype=amp):
                    logits = model(ids, mask)
                loss = loss_fn(logits.float(), y)
                if not torch.isfinite(loss):
                    if amp != torch.float16 or skipped >= 100:
                        raise FloatingPointError(f'loss = {loss.item()} на кроці {step}')
                    skipped += 1                  # fp16 overflow now and then: skip the batch, keep the schedule
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    bar.update(1)
                    continue
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(train_params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()   # .item() waits for the GPU
                step_times.append(time.time() - ts)
                bar.update(1)
                bar.set_postfix(loss=f'{ema:.4f}')
                if time.time() - last_save > args.save_every_min * 60:
                    checkpoint(epoch, step + 1)
                    last_save = time.time()
            bar.close()
            if out_of_time:
                say(f'Час вичерпано (--time-budget-h {args.time_budget_h}): епоху {epoch + 1} не завершено, '
                    f'зупиняюсь і беру найкращий результат.')
            else:
                step = len(batches)

            say('Оцінюю на val...')
            P = predict(model, seq_va, pad, dev, amp, eval_batch)
            thr = tune_threshold(Yva, P)
            m, _ = evaluate(Yva, P, thr, classes)
            score = (m['micro_f1'] + m['macro_f1']) / 2
            history.append({'epoch': epoch + 1, 'threshold': thr, **m, 'minutes': round((time.time() - t0) / 60, 1),
                            **({'partial_steps': step} if out_of_time else {})})
            show(f'  епоха {epoch + 1}, поріг {thr}', m)
            if score > best:
                best = score
                save_best(model, tok, out / 'best', {'base_model': args.model, 'max_len': args.max_len, 'threshold': thr,
                                                     'classes': classes, 'epoch': epoch + 1, 'val': m,
                                                     'lora': args.lora}, lora=bool(args.lora))
                np.save(out / 'val_probs.npy', P)
                say('  найкраща епоха поки що: модель збережено')
            if out_of_time:
                break
            checkpoint(epoch + 1, 0)
            start_step = 0
    except KeyboardInterrupt:
        if bar is not None:
            bar.close()
        checkpoint(epoch, step)
        keep_awake(False)
        say(f'\nЗупинено. Стан збережено (епоха {epoch + 1}, крок {step}). Щоб продовжити, запусти ту саму команду.')
        sys.exit(130)                             # non-zero, so a batch file stops here too
    except torch.cuda.OutOfMemoryError:
        if bar is not None:
            bar.close()
        keep_awake(False)
        say(f'\nНе вистачило пам\'яті відеокарти (батч {args.batch}). Закрий інші програми, що використовують '
            f'відеокарту, або запусти з меншим батчем, напр. --batch {max(1, args.batch // 2)} (і --fresh, якщо '
            f'навчання вже почалося).')
        sys.exit(1)

    # the best epoch: probabilities on test for step 5, metrics on val for the report
    gpu_peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev.type == 'cuda' else None
    model.to('meta')                              # free its memory (GPU and RAM) for the final model
    opt.state.clear()
    gc.collect()
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    say('\nЗавантажую найкращу епоху, рахую ймовірності для test...')
    cfg = json.loads((out / 'best' / 'config.json').read_text(encoding='utf-8'))
    if args.lora:
        enc = AutoModel.from_pretrained(args.model, dtype=load_dtype)
        add_lora(enc, args.lora, args.lora_alpha, 0.0)
        best_model = Classifier(enc, len(classes))
        load_trainable(best_model, torch.load(out / 'best' / 'lora.pt', map_location='cpu'))
        best_model.encoder = merge_lora(best_model.encoder)    # an ordinary model, as predict.py expects
        best_model.encoder.save_pretrained(out / 'best' / 'encoder')
        tok.save_pretrained(out / 'best' / 'encoder')
        torch.save(best_model.head.state_dict(), out / 'best' / 'head.pt')
        best_model.to(dev)
        Pva = predict(best_model, seq_va, pad, dev, amp, eval_batch)    # val again with the merged weights
        cfg['merge_max_diff'] = round(float(np.abs(Pva - np.load(out / 'val_probs.npy')).max()), 4)
        cfg['threshold'] = tune_threshold(Yva, Pva)
        save_json(out / 'best' / 'config.json', cfg)
        np.save(out / 'val_probs.npy', Pva)
        say(f'  LoRA злито з вагами моделі (ймовірності на val змінились щонайбільше на {cfg["merge_max_diff"]:.3f})')
    else:
        best_model = Classifier(AutoModel.from_pretrained(out / 'best' / 'encoder'), len(classes))
        best_model.head.load_state_dict(torch.load(out / 'best' / 'head.pt', map_location='cpu'))
        best_model.to(dev)
        Pva = np.load(out / 'val_probs.npy')
    np.save(out / 'test_probs.npy', predict(best_model, seq_te, pad, dev, amp, eval_batch))
    keep_awake(False)
    m, per = evaluate(Yva, Pva, cfg['threshold'], classes)
    by_origin = evaluate_by_origin(va, Yva, Pva, cfg['threshold'], classes)
    per_group = sorted(({'code': c, 'name': names.get(c, ''), 'train': int(Ytr[:, j].sum()), 'val': int(Yva[:, j].sum()),
                         'f1': None if np.isnan(per[j]) else round(float(per[j]), 4)} for j, c in enumerate(classes)),
                       key=lambda g: (g['f1'] is None, g['f1'] if g['f1'] is not None else 0))
    minutes = round((time.time() - t0) / 60, 1)
    save_json(report, {'model': args.model, 'args': vars(args), 'best_epoch': cfg['epoch'], 'threshold': cfg['threshold'],
                       'val': m, 'val_by_origin': by_origin, 'history': history, 'val_per_group': per_group,
                       'gpu_peak_gb': gpu_peak, 'skipped_fp16_steps': skipped, 'stopped_by_time_budget': out_of_time,
                       'minutes': minutes})
    if state_path.exists():
        state_path.unlink()                       # 1.3 GB, not needed once training has finished

    say(f'\nНайкраща епоха: {cfg["epoch"]}, поріг {cfg["threshold"]}')
    show('val', m)
    for o, x in by_origin.items():
        show(f'  {o} ({x["records"]})', x)
    worst = [g for g in per_group if g['f1'] is not None][:5]
    say('Найслабші групи (F1 на val): ' + '; '.join(f'{g["code"]} {g["f1"]:.2f}' for g in worst))
    if args.smoke:
        import shutil
        shutil.rmtree(out, ignore_errors=True)    # the smoke model is not needed
        sec = float(np.median(step_times[5:] or step_times))
        epoch_min = sec * ((n_train_full + args.batch - 1) // args.batch) / 60
        say(f'\nПеревірка пройшла за {minutes} хв; її модель видалено, звіт лишився.')
        if gpu_peak is not None:
            total_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            say(f'Пам\'ять відеокарти під час навчання: до {gpu_peak:.1f} ГБ із {total_gb:.1f} ГБ.')
        say(f'За швидкістю перевірки: повна епоха ≈ {epoch_min:.0f} хв, 3 епохи ≈ {3 * epoch_min / 60:.1f} год '
            f'(плюс оцінка на val після кожної епохи).')
        say('Повне навчання: python 03_train_transformer.py ' + ' '.join(a for a in sys.argv[1:] if a != '--smoke'))
        say(f'Звіт: {report}')
        return
    say(f'Модель: {out / "best"}')
    say(f'Звіт: {report}  ({minutes} хв)')


if __name__ == '__main__':
    main()
