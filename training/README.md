# Навчання, оцінка й передбачення

Скрипти пронумеровано в порядку запуску. Кожен має докладний опис на початку файлу (`python <скрипт> --help`
показує параметри). Повідомлення скриптів — українською, коментарі в коді — англійською.

| Скрипт | Що робить | Пише |
|---|---|---|
| `00_check_env.py` | Перевіряє Python, пам'ять, відеокарту, пакети; підказує команду встановлення PyTorch під вашу відеокарту; пробні кроки навчання з різними розмірами пакета | `reports/00_env.json` |
| `01_split.py` | Поділ на train / val / test (80 / 10 / 10) за джерелами, зі стратифікацією за 213 групами | `data/{train,val,test}.parquet`, `data/labels.json` |
| `02_baseline.py` | TF-IDF + логістична регресія (одна на групу) | `models/baseline*/` |
| `03_train_transformer.py` | Донавчання BERT-подібного кодувальника (за замовчуванням SPECTER2; будь-яка модель Hugging Face через `--model`) | `models/<назва>/` |
| `04_ensemble.py` | Поєднання моделей: ваги й поріг підбирає перехресною перевіркою всередині val | `models/ensemble*/` |
| `05_evaluate.py` | Одноразова оцінка на test, порівняння з попередніми системами | `reports/05_test*.json`, `.xlsx` |
| `06_clean_labels.py` | Застосовує `06_label_fixes.csv` до train: 933 записи виправлено, 916 виключено | `data/train_clean.parquet` |
| `07_kaggle_package.py` | Архів для Kaggle: дані й скрипт навчання (~95 МБ) | `kaggle_upload.zip` |
| `kaggle_train.ipynb` | Ноутбук Kaggle: дві великі моделі паралельно, кожна на своїй T4 | `<модель>.zip` в Output |
| `08_import_kaggle.py` | Переносить моделі, завантажені з Kaggle, у `models/` і `reports/` | `models/`, `reports/` |
| `predict.py` | Групи для нових публікацій (CSV, Excel або один запис з командного рядка) | CSV / Excel |
| `common.py` | Спільне: шляхи, мітки, метрики, підбір порогу | — |

## Де мають лежати дані

```
anzsrc-for-classifier/
├── anzsrc4_cleaned.csv     ← набір даних (результат data_prep, у репозиторії його немає)
├── anzsrc2020_for.xlsx     ← назви груп і розділів (ABS)
└── training/
    ├── data/  models/  reports/     ← створюють скрипти
    └── ...
```

## Як було отримано підсумковий результат

Windows, Anaconda Prompt, тека `training`:

```bat
python 00_check_env.py
python 01_split.py
python 02_baseline.py
python 03_train_transformer.py                                           :: SPECTER2, ~2,2 год
python 03_train_transformer.py --model allenai/scibert_scivocab_uncased  :: SciBERT, ~2,2 год
python 04_ensemble.py --models baseline specter2
python 05_evaluate.py

run_clean_training.bat        :: 06_clean_labels + TF-IDF, SPECTER2, RoBERTa-base на очищених мітках (~5–6 год)

python 07_kaggle_package.py   :: далі Kaggle, див. нижче
run_after_kaggle.bat          :: 08_import_kaggle + поєднання п'яти моделей + test

run_final_ensemble.bat        :: TF-IDF + SPECTER2 + ModernBERT-large → models\ensemble_final
```

Перерване навчання (`Ctrl+C`, вимкнення живлення) продовжується тією самою командою: стан зберігається кожні
20 хвилин і після кожної епохи (`models/<назва>/last.pt`). Під час навчання скрипт просить Windows не
засинати.

### Великі моделі на Kaggle

1. `python 07_kaggle_package.py` → `kaggle_upload.zip`.
2. kaggle.com → **Datasets → New Dataset** (приватний) → завантажити архів.
3. **Code → New Notebook → File → Import Notebook** → `kaggle_train.ipynb`; праворуч: **Add Input** → цей
   набір даних, **Accelerator** → GPU T4 x2, **Internet** → On.
4. **Save Version → Save & Run All (Commit)**. Браузер можна закрити. ModernBERT-large навчався 7,0 год,
   DeBERTa-v3-large — 10,4 год (обидва паралельно; навчання саме зупиняється не пізніше ніж за 11 год).
5. У версії ноутбука: **Output** → завантажити `<модель>.zip` і `reports_kaggle.zip`, потім
   `run_after_kaggle.bat` (шукає архіви в теці «Завантаження»).

На T4 немає bf16, тож там навчання йде у fp16 (скрипт вибирає сам і пропускає кроки з нескінченними
втратами; в обох запусках таких кроків не було).

### Без хмари: велика модель з LoRA на 4 ГБ

`run_large_training.bat` навчає ModernBERT-large з LoRA (rank 32): кодувальник заморожено в bf16, навчаються
лише низькорангові додатки, наприкінці вони вливаються у ваги. Займає ~2,5 ГБ відеопам'яті, 7–8 год на
RTX 3050 Ti Laptop. Це запасний шлях: результати в README отримано повним донавчанням на Kaggle.

## Передбачення для нових публікацій

```bat
python predict.py --title "Soil carbon under no-till farming" --abstract "We measured ..."
python predict.py --input new_papers.xlsx --output predictions.xlsx
python predict.py --input new_papers.csv --title-col Title --abstract-col Abstract --output predictions.csv
python predict.py --input new_papers.xlsx --output top3.xlsx --threshold 0 --max-labels 3
```

За замовчуванням береться найновіше збережене поєднання (`ensemble_final`, потім `ensemble_large`, …), тож
потрібні `models/ensemble_final/` і три моделі, які воно називає (`baseline_clean`, `specter2_clean`,
`modernbert_large_clean`), а також `data/labels.json`. До вхідних стовпців додаються:

| Стовпець | Зміст |
|---|---|
| `pred_codes` | групи, які призначає модель, напр. `3103;3104` (щонайменше одна) |
| `pred_groups` | ті самі групи з назвами |
| `divisions` | їхні розділи |
| `top5` | п'ять найімовірніших груп з імовірностями |
| `max_probability` | імовірність найімовірнішої групи: що нижча, то менш певна модель |

`--threshold` змінює поріг (вищий — менше, але точніших груп), `--max-labels` обмежує кількість груп.
Режим впевненості з [docs/results.md](../docs/results.md#режим-впевненості): записи з
`max_probability` ≥ 0,905 модель розмічає сама (89,6 % правильних на test), решту — людина, з `top5` як
підказкою.

Моделей у репозиторії немає (разом кілька гігабайтів): їх треба навчити цими скриптами.
