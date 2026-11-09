#!/usr/bin/env python3
"""
ANZSRC FoR 2020 classifier, step 0: check the environment.

    python 00_check_env.py

Run it from the training folder (next to ../anzsrc4_cleaned.csv), inside the project's conda environment.

1. Python, CPU, RAM, free disk space, the NVIDIA GPU and its driver, installed packages.
2. The dataset ../anzsrc4_cleaned.csv: rows, groups, empty fields.
3. PyTorch. If it is missing, or has no CUDA while an NVIDIA GPU is present, the script looks up which
   PyTorch builds exist for this GPU, driver and Python and prints the pip command to install one.
   Run that command, then run this script again.
4. With PyTorch working it downloads SPECTER2 (allenai/specter2_base, ~440 MB, kept in the Hugging Face
   cache), measures text length in tokens and runs a few training steps on real abstracts with several
   batch sizes: what fits in GPU memory and how long an epoch would take. Without an NVIDIA GPU it
   measures how fast the CPU encodes texts instead.

Results are printed and saved to reports/00_env.json.
"""
import gc
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from importlib import metadata
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'anzsrc4_cleaned.csv'
REPORT = ROOT / 'reports' / '00_env.json'
MODEL_NAME = 'allenai/specter2_base'
N_LABELS = 213
N_ALL = 109_547
N_TRAIN = 87_600            # about 80 % of the records
EPOCHS = 3
MAX_LENS = (512, 384)
BATCHES = (4, 8, 16, 32)
PACKAGES = ['numpy', 'scipy', 'pandas', 'pyarrow', 'scikit-learn', 'openpyxl', 'tqdm', 'psutil', 'transformers']
EXTRA = ['tokenizers', 'huggingface_hub', 'safetensors', 'torch']


def say(*a):
    print(*a, flush=True)


def run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def vtuple(s):
    try:
        return tuple(int(x) for x in s.split('.')[:2])
    except Exception:
        return None


# ---------------------------------------------------------------- system
def system_info():
    s = {'python': platform.python_version(), 'executable': sys.executable, 'os': platform.platform(),
         'conda_env': os.environ.get('CONDA_DEFAULT_ENV', ''), 'cpu': platform.processor(), 'cpu_threads': os.cpu_count()}
    if sys.platform == 'win32':
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            s['cpu'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    try:
        import psutil
        vm = psutil.virtual_memory()
        s.update(cpu_cores=psutil.cpu_count(logical=False), ram_gb=round(vm.total / 2**30, 1),
                 ram_free_gb=round(vm.available / 2**30, 1))
    except ImportError:
        pass
    for key, path in (('disk_free_gb_project', ROOT), ('disk_free_gb_home', Path.home())):
        try:
            s[key] = round(shutil.disk_usage(path).free / 2**30, 1)
        except OSError:
            pass
    return s


def package_versions():
    v = {}
    for p in PACKAGES + EXTRA:
        try:
            v[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            v[p] = None
    return v


def nvidia_info():
    exe = shutil.which('nvidia-smi')
    if not exe and sys.platform == 'win32':
        for p in (r'C:\Windows\System32\nvidia-smi.exe', r'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe'):
            if Path(p).exists():
                exe = p
                break
    if not exe:
        return {'gpus': [], 'driver_cuda': None}
    q = '--format=csv,noheader,nounits'
    out = run([exe, '--query-gpu=name,memory.total,driver_version,compute_cap', q])
    if out is None:
        out = run([exe, '--query-gpu=name,memory.total,driver_version', q])
    gpus = []
    for line in (out or '').strip().splitlines():
        f = [x.strip() for x in line.split(',')]
        g = {'name': f[0], 'driver': f[2] if len(f) > 2 else None}
        try:
            g['memory_gb'] = round(float(f[1]) / 1024, 1)
        except (ValueError, IndexError):
            pass
        try:
            g['compute_cap'] = float(f[3])
        except (ValueError, IndexError):
            pass
        gpus.append(g)
    m = re.search(r'CUDA Version:\s*([\d.]+)', run([exe]) or '')
    return {'gpus': gpus, 'driver_cuda': m.group(1) if m else None}


def other_video_adapters():
    if sys.platform != 'win32':
        return []
    out = run(['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_VideoController).Name'])
    return [x.strip() for x in (out or '').splitlines() if x.strip()]


# ---------------------------------------------------------------- PyTorch build
def torch_versions(tag):
    """torch versions published for this Python and platform at https://download.pytorch.org/whl/<tag>"""
    py = f'cp{sys.version_info.major}{sys.version_info.minor}'
    plat = r'win_amd64' if sys.platform == 'win32' else r'manylinux[\w.]*_x86_64'
    try:
        html = urllib.request.urlopen(f'https://download.pytorch.org/whl/{tag}/torch/', timeout=30).read()
    except Exception:
        return None
    found = set(re.findall(rf'torch-(\d+\.\d+\.\d+)(?:\+|%2B){tag}-{py}-{py}-{plat}\.whl', html.decode('utf-8', 'replace')))
    return sorted(found, key=lambda v: tuple(int(x) for x in v.split('.')))


def recommend_torch(nv):
    gpus = nv.get('gpus') or []
    if not gpus:
        tags = ['cpu']
    else:
        cc, drv = gpus[0].get('compute_cap'), vtuple(nv.get('driver_cuda') or '')
        cands = []
        if cc is None or cc >= 7.5:                       # Turing (RTX 20xx, GTX 16xx) and newer
            cands += [('cu130', (13, 0)), ('cu129', (12, 9)), ('cu128', (12, 8))]
        cands += [('cu126', (12, 6))]                     # also Maxwell, Pascal, Volta (GTX 9xx/10xx)
        tags = [t for t, need in cands if drv is None or need <= drv]
    rec = {'checked': {}}
    best = None
    for k, tag in enumerate(tags):                        # newest PyTorch wins; on a tie the newer CUDA (listed first)
        vs = torch_versions(tag)
        rec['checked'][tag] = vs[-3:] if vs else vs
        if vs:
            key = (tuple(int(x) for x in vs[-1].split('.')), -k)
            if best is None or key > best[0]:
                best = (key, tag, vs[-1])
    if best:
        rec.update(tag=best[1], version=best[2])
    elif tags:
        rec['tag'] = tags[0]                              # index not reachable: best guess
    if 'tag' in rec:
        rec['command'] = f'python -m pip install torch --index-url https://download.pytorch.org/whl/{rec["tag"]}'
    return rec


# ---------------------------------------------------------------- data
def dataset_info():
    info = {'path': str(DATA), 'found': DATA.exists()}
    if not info['found']:
        return info, None
    import pandas as pd
    d = pd.read_csv(DATA, dtype=str, keep_default_na=False, encoding='utf-8-sig',
                    usecols=['Record_ID', 'Title', 'Abstract', 'ANZSRC_4D'])
    labels = d.ANZSRC_4D.str.split(';')
    groups = {c for s in labels for c in s if c}
    info.update(size_mb=round(DATA.stat().st_size / 2**20, 1), rows=len(d), groups=len(groups),
                empty_labels=int(d.ANZSRC_4D.eq('').sum()), empty_abstracts=int(d.Abstract.str.strip().eq('').sum()),
                duplicate_ids=int(d.Record_ID.duplicated().sum()),
                labels_per_record=round(float(labels.str.len().mean()), 2))
    return info, d


def sample_texts(d, sep, n=3000):
    if d is None:
        return ['Soil carbon and land management in tropical forests of northern Australia. ' * 40] * 64
    idx = random.Random(0).sample(range(len(d)), min(n, len(d)))
    return [f'{d.Title.iat[i]}{sep}{d.Abstract.iat[i]}' for i in idx]


def token_stats(tok, texts):
    import numpy as np
    n = np.array([len(x) for x in tok(texts, add_special_tokens=True, truncation=False)['input_ids']])
    st = {'sample': int(len(n)), 'median': int(np.median(n)), 'p90': int(np.percentile(n, 90)),
          'p95': int(np.percentile(n, 95)), 'max': int(n.max())}
    for L in MAX_LENS:
        st[f'over_{L}'] = round(float((n > L).mean()), 3)
        st[f'mean_tokens_at_{L}'] = round(float(np.minimum(n, L).mean()), 1)
    return st


# ---------------------------------------------------------------- model trials
def load_model(torch):
    from transformers import AutoModel
    enc = AutoModel.from_pretrained(MODEL_NAME)

    class Net(torch.nn.Module):          # encoder + one linear layer over the [CLS] vector, as in step 3
        def __init__(self):
            super().__init__()
            self.enc = enc
            self.head = torch.nn.Linear(enc.config.hidden_size, N_LABELS)

        def forward(self, ids, mask):
            return self.head(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])

    return Net()


def train_trial(torch, tok, texts, st, device='cuda', checkpointing=False, batches=BATCHES):
    """a few real training steps per (max_len, batch); stops at the first batch that does not fit.
    checkpointing: recompute activations in the backward pass (less GPU memory, ~30 % slower)"""
    dev = torch.device(device)
    cuda = dev.type == 'cuda'
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30 if cuda else None
    bf16 = torch.cuda.is_bf16_supported() if cuda else True
    dtype = torch.bfloat16 if bf16 else torch.float16
    model = load_model(torch)
    if checkpointing:
        model.enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=cuda and not bf16)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    out = {'precision': 'bf16' if bf16 else 'fp16', 'gpu_memory_gb': round(total, 1) if total else None,
           'checkpointing': checkpointing, 'runs': []}
    for L in MAX_LENS:
        best_speed = 0.0
        for bs in batches:
            enc = tok(texts[:bs], truncation=True, max_length=L, padding='max_length', return_tensors='pt')
            ids, mask = enc['input_ids'].to(dev), enc['attention_mask'].to(dev)
            y = torch.zeros(bs, N_LABELS, device=dev)
            y[:, 0] = 1
            row = {'max_len': L, 'batch': bs}
            logits = loss = None
            if cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            try:
                times = []
                for _ in range(7):
                    sync()
                    t0 = time.perf_counter()
                    with torch.autocast(dev.type, dtype=dtype):
                        logits = model(ids, mask)
                    loss = loss_fn(logits.float(), y)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    sync()
                    times.append(time.perf_counter() - t0)
                sec = sorted(times[2:])[2]                        # median of the 5 timed steps
                row.update(sec_per_step=round(sec, 3), texts_per_sec=round(bs / sec, 1), fits=True)
                if cuda:
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    row['peak_gb'] = round(peak, 2)
                    # Windows drivers may spill into system RAM instead of failing: treat as not fitting
                    if peak > 0.9 * total or bs / sec < 0.8 * best_speed:
                        row.update(fits=False, error='close to the memory limit or slowed down')
                best_speed = max(best_speed, bs / sec) if row['fits'] else best_speed
            except torch.cuda.OutOfMemoryError:
                row.update(fits=False, error='out of memory')
            logits = loss = None
            del ids, mask, y
            opt.zero_grad(set_to_none=True)
            gc.collect()
            if cuda:
                torch.cuda.empty_cache()
            out['runs'].append(row)
            if not row['fits']:
                break
        ok = [r for r in out['runs'] if r['max_len'] == L and r['fits']]
        if ok:
            b = max(ok, key=lambda r: r['texts_per_sec'])
            frac = st[f'mean_tokens_at_{L}'] / L if st else 1.0
            minutes = N_TRAIN / b['batch'] * b['sec_per_step'] * frac * 1.1 / 60      # +10 % for validation
            out[f'best_{L}'] = {'batch': b['batch'], 'epoch_minutes': round(minutes, 1),
                                f'hours_{EPOCHS}_epochs': round(minutes * EPOCHS / 60, 1)}
    del model, opt
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return out


def cpu_trial(torch, tok, texts, st):
    """encoding speed on the CPU (the route without an NVIDIA GPU: fixed text vectors + a light classifier)"""
    model = load_model(torch).eval()
    enc = tok(texts[:16], truncation=True, max_length=512, padding='max_length', return_tensors='pt')
    with torch.inference_mode():
        model(enc['input_ids'], enc['attention_mask'])
        t0 = time.perf_counter()
        for _ in range(2):
            model(enc['input_ids'], enc['attention_mask'])
    sec = (time.perf_counter() - t0) / 32
    frac = st['mean_tokens_at_512'] / 512 if st else 1.0
    return {'threads': torch.get_num_threads(), 'sec_per_text_512': round(sec, 3),
            'hours_encode_all': round(sec * frac * N_ALL / 3600, 1),
            'hours_finetune_epoch_estimate': round(3 * sec * frac * N_TRAIN / 3600, 1)}


# ---------------------------------------------------------------- main
def main():
    rep = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'errors': []}
    try:
        check(rep)
    except Exception as e:                                 # keep whatever was collected
        rep['errors'].append(f'{type(e).__name__}: {e}')
        say(f'\nПОМИЛКА: {type(e).__name__}: {e}')
    finally:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
        say(f'\nЗвіт збережено: {REPORT}')


def check(rep):
    say('=== Крок 0: перевірка середовища ===\n')
    s = rep['system'] = system_info()
    say(f'Python {s["python"]}  ({s["executable"]})')
    if s['conda_env'] != 'anzsrc' and 'anzsrc' not in Path(sys.executable).parts:
        say(f'  Увага: активне середовище «{s["conda_env"] or "не conda"}», а не anzsrc. Виконай: conda activate anzsrc')
    if vtuple(s['python']) < (3, 10):
        say('  Увага: потрібен Python 3.10 або новіший.')
    say(f'Процесор: {s["cpu"]}; ядер {s.get("cpu_cores", "?")}, потоків {s["cpu_threads"]}')
    say(f'Пам\'ять: {s.get("ram_gb", "?")} ГБ (вільно {s.get("ram_free_gb", "?")} ГБ)')
    say(f'Вільне місце: диск проєкту {s.get("disk_free_gb_project", "?")} ГБ, '
        f'домашній диск {s.get("disk_free_gb_home", "?")} ГБ')

    pk = rep['packages'] = package_versions()
    missing = [p for p in PACKAGES if not pk[p]]
    say('Пакети: ' + ', '.join(f'{p} {pk[p]}' for p in PACKAGES + EXTRA if pk[p]))
    if missing:
        say(f'  Бракує: {", ".join(missing)}. Встанови: python -m pip install {" ".join(missing)}')

    nv = rep['nvidia'] = nvidia_info()
    if nv['gpus']:
        for g in nv['gpus']:
            say(f'Відеокарта: {g["name"]}, {g.get("memory_gb", "?")} ГБ, драйвер {g["driver"]}, '
                f'CUDA драйвера до {nv["driver_cuda"]}, compute capability {g.get("compute_cap", "?")}')
    else:
        rep['video_adapters'] = other_video_adapters()
        say('Відеокарту NVIDIA не знайдено' +
            (f' (є: {"; ".join(rep["video_adapters"])})' if rep['video_adapters'] else ''))

    d = None
    if pk['pandas']:
        say('\nЧитаю датасет...')
        info, d = dataset_info()
        rep['dataset'] = info
        if not info['found']:
            say(f'  Не знайдено {DATA}. Поклади anzsrc4_cleaned.csv у папку ANZSRC, поруч із папкою training.')
        else:
            rows = f'{info["rows"]:,}'.replace(',', ' ')
            say(f'  {rows} записів, {info["groups"]} груп, у середньому {info["labels_per_record"]} мітки на запис; '
                f'порожніх міток {info["empty_labels"]}, порожніх анотацій {info["empty_abstracts"]}, '
                f'повторів ID {info["duplicate_ids"]}')

    try:
        import torch
    except ImportError:
        torch = None
    cuda_ok = bool(torch) and torch.cuda.is_available()
    if torch:
        rep['torch'] = {'version': torch.__version__, 'cuda_build': torch.version.cuda, 'cuda_available': cuda_ok}
        say(f'\nPyTorch {torch.__version__}, збірка CUDA: {torch.version.cuda or "немає (CPU)"}, '
            f'відеокарта доступна: {"так" if cuda_ok else "ні"}')
    if not torch or (nv['gpus'] and not cuda_ok):
        say('\nШукаю збірку PyTorch для цього комп\'ютера...')
        rec = rep['torch_install'] = recommend_torch(nv)
        if 'command' not in rec:
            say(f'  Драйвер NVIDIA застарий (CUDA {nv["driver_cuda"]}): онови його з nvidia.com і запусти скрипт ще раз.')
            return
        say(f'  Підходить збірка {rec["tag"]}' + (f', PyTorch {rec["version"]}' if rec.get('version') else
                                                   ' (сайт PyTorch не відповів, це найімовірніший варіант)'))
        say('\nВстанови PyTorch і запусти цей скрипт ще раз:')
        if torch:
            say('  python -m pip uninstall -y torch')
        say(f'  {rec["command"]}')
        if not nv['gpus']:
            say('\nБез відеокарти NVIDIA дотренування трансформера на процесорі триватиме дні; '
                'другий запуск покаже, скільки займе альтернативний шлях.')
        return
    if missing:
        say('\nСпершу встанови пакети, яких бракує, і запусти скрипт ще раз.')
        return

    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    say(f'\nЗавантажую {MODEL_NAME} (перший раз ~440 МБ)...')
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        rep['errors'].append(f'model download: {e}')
        say(f'  Не вдалося завантажити модель з Hugging Face: {e}')
        return
    texts = sample_texts(d, tok.sep_token)
    st = rep['tokens'] = token_stats(tok, texts) if d is not None else None
    if st:
        say(f'Довжина тексту (назва + анотація) у токенах: медіана {st["median"]}, 90 % — до {st["p90"]}, '
            f'довших за 512 — {st["over_512"]:.1%}, за 384 — {st["over_384"]:.1%}')

    if cuda_ok:
        name = torch.cuda.get_device_name(0)
        small = torch.cuda.get_device_properties(0).total_memory / 2**30 < 7.5
        modes = (False, True) if small else (False,)        # small GPU: also try gradient checkpointing
        trials = rep['gpu_trial'] = []
        for ckpt in modes:
            say(f'\nПробне навчання на відеокарті ({name})' + (', з економією пам\'яті (gradient checkpointing)' if ckpt else '')
                + '...')
            tr = train_trial(torch, tok, texts, st, checkpointing=ckpt, batches=(2,) + BATCHES if small else BATCHES)
            trials.append(tr)
            for L in MAX_LENS:
                runs = [r for r in tr['runs'] if r['max_len'] == L]
                if not runs:
                    continue
                say(f'  max_len {L}: ' + '; '.join(
                    f'батч {r["batch"]}: ' + (f'{r["sec_per_step"]} с/крок, {r.get("peak_gb", "?")} ГБ' if r['fits']
                                              else 'не вміщається') for r in runs))
                if f'best_{L}' in tr:
                    b = tr[f'best_{L}']
                    say(f'    → батч {b["batch"]}: ≈ {b["epoch_minutes"]} хв на епоху, '
                        f'{EPOCHS} епохи ≈ {b[f"hours_{EPOCHS}_epochs"]} год')
        options = [(tr[f'best_{L}'][f'hours_{EPOCHS}_epochs'], L, tr['checkpointing'], tr[f'best_{L}']['batch'])
                   for L in MAX_LENS for tr in trials if f'best_{L}' in tr]
        best = min([o for o in options if o[1] == 512] or options or [None], key=lambda o: o[0] if o else 0)
        rep['gpu_best'] = dict(zip(('hours', 'max_len', 'checkpointing', 'batch'), best)) if best else None
        laptop = ' На ноутбуці під довгим навантаженням може бути на 10–30 % повільніше.' if 'laptop' in name.lower() else ''
        if not best:
            say('\nВисновок: модель не вміщається у відеопам\'ять навіть з найменшим батчем.')
        else:
            how = (f'max_len {best[1]}, батч {best[3]}' + (', gradient checkpointing' if best[2] else ''))
            verdict = 'реальне' if best[0] <= 8 else 'можливе, але довге'
            say(f'\nВисновок: дотренування трансформера на цій відеокарті {verdict}: {how}, '
                f'{EPOCHS} епохи ≈ {best[0]} год.{laptop}')
    else:
        say('\nВідеокарти немає: міряю швидкість на процесорі...')
        ct = rep['cpu_trial'] = cpu_trial(torch, tok, texts, st)
        say(f'  {ct["sec_per_text_512"]} с на текст: вектори для всього датасету ≈ {ct["hours_encode_all"]} год, '
            f'одна епоха дотренування ≈ {ct["hours_finetune_epoch_estimate"]} год')


if __name__ == '__main__':
    main()
