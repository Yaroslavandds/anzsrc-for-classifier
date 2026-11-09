#!/usr/bin/env python3
"""
ANZSRC FoR 2020 classifier, step 7: a package for training the large models on Kaggle.

    python 07_kaggle_package.py

Makes kaggle_upload.zip (about 95 MB) next to this script: common.py, 03_train_transformer.py and from data/
train_clean.parquet, val.parquet, test.parquet, labels.json. On Kaggle it becomes a private dataset that the notebook
kaggle_train.ipynb reads. Nothing else leaves the computer. The results come back with 08_import_kaggle.py.
"""
import zipfile

from common import DATA, ROOT, say

FILES = [ROOT / 'common.py', ROOT / '03_train_transformer.py'] + \
        [DATA / f for f in ('train_clean.parquet', 'val.parquet', 'test.parquet', 'labels.json')]


def main():
    missing = [str(f) for f in FILES if not f.exists()]
    if missing:
        raise SystemExit('Немає файлів: ' + ', '.join(missing) + '. Для train_clean.parquet спершу запусти 06_clean_labels.py.')
    out = ROOT / 'kaggle_upload.zip'
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_STORED) as z:     # parquet is compressed already
        for f in FILES:
            z.write(f, f.relative_to(ROOT).as_posix())
    say(f'Готово: {out} ({out.stat().st_size / 2 ** 20:.0f} МБ)')
    for f in FILES:
        say(f'  {f.relative_to(ROOT).as_posix()}')
    say('\nДалі: на kaggle.com створи приватний набір даних (Datasets → New Dataset) і завантаж туди цей файл.')


if __name__ == '__main__':
    main()
