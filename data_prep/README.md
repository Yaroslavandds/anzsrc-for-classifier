# Підготовка даних

Вісім скриптів, які з об'єднаного набору (118 099 записів) роблять навчальний набір `anzsrc4_cleaned.csv`
(109 547 записів). Кожен крок читає результат попереднього, вхідних файлів не змінює й пише в окрему теку
результат, журнал змін (`*.pkl`) і підсумок (`summary.json`). Рішення, ухвалені після ручного перегляду
(групи однакових анотацій, серії, службові записи, хибні журнальні коди), записані прямо в скриптах, тож
повторний прогін дає той самий результат.

Докладно про кожен крок, поля й походження міток: [docs/data.md](../docs/data.md).

| Крок | Скрипт | Що робить | Записів після |
|---|---|---|---:|
| 1 | `step1_dedup.py` | Дублікати: той самий DOI (і його версії `.v1`, `.v2`), однакові назва й анотація, варіант назви з тією самою анотацією, майже однакові тексти (TF-IDF, косинус ≥ 0,90). Лишається запис зі справжніми (не ШІ) мітками, з анотацією, основна публікація | 117 933 |
| 2 | `step2_shared_abstracts.py` | Одна анотація під різними назвами: кожну групу переглянуто вручну (прибрати анотацію, видалити записи або залишити один) | 117 521 |
| 3 | `step3_title_only.py` | Записи без придатної анотації: службові (виправлення, редакційні, некрологи) видалено, решту винесено в `title_only.csv` | 114 850 |
| 4 | `step4_junk_abstracts.py` | Службові записи з анотацією, спам, «анотації» без змісту (повтор назви, бібліографічне посилання) | 114 566 |
| 5 | `step5_labels.py` | Мітки: узгодження `ANZSRC_4D` з полем `ANZSRC`, записи без тематичної інформації, записи з понад 6 групами | 114 160 |
| 6 | `step6_for2008_language.py` | Коди FoR 2008 перевірено за офіційною таблицею відповідності ABS; записи не англійською винесено в `non_english.csv` | 113 583 |
| 7 | `step7_cleanup.py` | Записи, що не описують дослідження; 130 груп майже однакових текстів (переглянуто вручну); очищення тексту (HTML, зіпсоване кодування, службові рядки, посилання) | 112 860 |
| 8 | `step8_labels.py` | Мітки, яких не підтримує текст: хибні журнальні коди; перехресна перевірка TF-IDF + LinearSVC (5 частин за джерелами), сумнівні записи винесено в `disputed_labels.csv` | **109 547** |

Побічні файли після всіх кроків: `title_only.csv` — 2 579 записів без анотації, `non_english.csv` — 701,
`disputed_labels.csv` — 3 313 (мітки не змінено, пропозиції моделі — в окремих стовпцях). До навчання вони не
входять.

## Запуск

З кореня репозиторію (вхідний CSV — 26 стовпців, див. [docs/data.md](../docs/data.md#поля)):

```bat
python data_prep\step1_dedup.py merged_dataset.csv work\step1
python data_prep\step2_shared_abstracts.py work\step1\dataset_step1_dedup.csv work\step2
python data_prep\step3_title_only.py work\step2\dataset_step2.csv work\step3
python data_prep\step4_junk_abstracts.py work\step3\dataset_step3.csv work\step3\title_only.csv work\step4
python data_prep\step5_labels.py work\step4\dataset_step4.csv work\step4\title_only.csv work\step5
python data_prep\step6_for2008_language.py work\step5\dataset_step5.csv work\step5\title_only.csv data_prep\anzsrc_for2008_to_2020_ABS.csv work\step6
python data_prep\step7_cleanup.py work\step6\dataset_step6.csv work\step6\title_only.csv work\step6\non_english.csv work\step7
python data_prep\step8_labels.py work\step7\dataset_step7.csv work\step8
copy work\step8\dataset_step8.csv anzsrc4_cleaned.csv
```

Потрібні `pandas`, `numpy`, `scikit-learn` ≥ 1.3, `ftfy` (крок 7) і `pycld2` (крок 6). Крок 8 триває близько
20 хвилин на двох ядрах, решта — від секунд до кількох хвилин. Тека `work/` у `.gitignore`.

## Правила, яких дотримано на всіх кроках

* Справжні мітки (з вихідного набору, репозиторіїв, грантових баз) мають перевагу над мітками від ШІ-моделей;
  серед дублікатів першим видаляється запис з ШІ-мітками.
* Передбачення моделей ніколи не записуються як мітки. Мітки лише знімаються, коли їх не підтримує текст або
  офіційна таблиця відповідності; додаються тільки групи, які однозначно випливають із поля `ANZSRC` (напряму
  або через таблицю ABS).
* Поле `ANZSRC` (коди в тому вигляді, як їх подало джерело) ніде не змінюється.
* Крім задокументованих правок (`Abstract` і `Has_usable_abstract` для шаблонних анотацій, `ANZSRC_4D` на
  кроках 5, 6 і 8, текст `Title` і `Abstract` на кроці 7), записи переносяться без змін і в тому самому порядку.

## Файл ABS

`anzsrc_for2008_to_2020_ABS.csv` — таблиця відповідності FoR 2008 → FoR 2020 (стовпці `code_2008`, `code_2020`,
`partial`) з публікації Australian Bureau of Statistics «1297.0 ANZSRC 2020 — Correspondence Tables», Table 2
(випуск 24 жовтня 2025 р.), ліцензія [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
