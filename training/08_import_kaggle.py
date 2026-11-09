#!/usr/bin/env python3
"""
ANZSRC FoR 2020 classifier, step 8: bring the models trained on Kaggle into models/ and reports/.

    python 08_import_kaggle.py                      # looks in your Downloads folder
    python 08_import_kaggle.py "D:\\some\\folder"     # or in another folder, or give the .zip files themselves

Takes the files downloaded from the notebook's Output: <model>.zip (e.g. deberta_large_clean.zip,
modernbert_large_clean.zip) and reports_kaggle.zip. A zip of the whole Output (with these zips inside) works too.
A model folder that already exists in models/ is kept as <name>_previous.
"""
import io
import shutil
import sys
import time
import zipfile
from pathlib import Path

from common import MODELS, REPORTS, say

RECENT_DAYS = 14


def model_dirs(z):
    """top-level folders in the zip that hold a finished model"""
    return sorted({n.split('/')[0] for n in z.namelist() if n.count('/') == 1 and n.endswith('/test_probs.npy')})


def extract_model(z, name):
    target = MODELS / name
    tmp = MODELS / f'{name}.importing'
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for info in z.infolist():
        if info.filename.startswith(name + '/') and not info.is_dir():
            dest = tmp / info.filename[len(name) + 1:]
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dest, 'wb') as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    if target.exists():
        old = MODELS / f'{name}_previous'
        shutil.rmtree(old, ignore_errors=True)
        target.rename(old)
        say(f'  попередню models/{name} перейменовано на {old.name}')
    tmp.rename(target)
    size = sum(f.stat().st_size for f in target.rglob('*') if f.is_file()) / 2 ** 30
    say(f'  models/{name}: {size:.2f} ГБ')


def extract_reports(z):
    n = 0
    for info in z.infolist():
        parts = info.filename.split('/')
        if len(parts) == 2 and parts[0] == 'reports' and parts[1].startswith('03_') and parts[1].endswith('.json'):
            REPORTS.mkdir(parents=True, exist_ok=True)
            (REPORTS / parts[1]).write_bytes(z.read(info))
            n += 1
    return n


def handle(z, label, found):
    for name in model_dirs(z):
        say(f'{label}: модель {name}')
        extract_model(z, name)
        found['models'].append(name)
    n = extract_reports(z)
    if n:
        say(f'{label}: звіти кроку 3 ({n})')
        found['reports'] += n
    for inner in [i for i in z.infolist() if i.filename.lower().endswith('.zip')]:      # zip of the whole Output
        with z.open(inner) as f:
            data = io.BytesIO(f.read()) if inner.file_size < 4 * 2 ** 30 else None
        if data is not None:
            with zipfile.ZipFile(data) as zz:
                handle(zz, f'{label} → {inner.filename}', found)


def main():
    args = [Path(a) for a in sys.argv[1:]] or [Path.home() / 'Downloads']
    zips = []
    for a in args:
        if a.is_dir():
            cutoff = time.time() - RECENT_DAYS * 86400
            zips += sorted(p for p in a.glob('*.zip') if p.stat().st_mtime > cutoff)
        elif a.suffix.lower() == '.zip' and a.exists():
            zips.append(a)
        else:
            raise SystemExit(f'Не знайдено: {a}')
    found = {'models': [], 'reports': 0}
    for p in zips:
        try:
            with zipfile.ZipFile(p) as z:
                names = z.namelist()
                if any(n.endswith('/test_probs.npy') or n.startswith('reports/03_') or n.lower().endswith('.zip')
                       for n in names):
                    handle(z, p.name, found)
        except zipfile.BadZipFile:
            continue
    if not found['models']:
        raise SystemExit('Не знайшов жодної моделі з Kaggle. Завантаж з Output ноутбука файли <модель>.zip '
                         '(напр. deberta_large_clean.zip) у теку «Завантаження» або вкажи теку: '
                         'python 08_import_kaggle.py "шлях"')
    say(f'\nГотово: {", ".join(found["models"])}. Далі: 04_ensemble.py і 05_evaluate.py (або run_after_kaggle.bat).')


if __name__ == '__main__':
    main()
