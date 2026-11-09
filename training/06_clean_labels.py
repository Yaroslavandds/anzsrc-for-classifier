#!/usr/bin/env python3
"""
ANZSRC FoR 2020 classifier, step 6: cleaner labels for training.

    python 06_clean_labels.py

Applies 06_label_fixes.csv (next to this script) to data/train.parquet and writes data/train_clean.parquet
(the part used for training) and data/train_excluded.parquet (records taken out, with the reason).
Val and test are not touched, so models trained on train_clean are compared with the old ones on the same records.
Every change, for review: reports/06_label_fixes.xlsx; summary: reports/06_clean_labels.json. A few seconds.

How the fixes were found (all on train, with an out-of-fold TF-IDF model: 5 folds grouped by source, so every
record was scored by a model that had seen neither it nor its journal):
* journal codes: 16 journals where one code sits on at least 80 % of the articles but their texts do not
  support it, checked by hand on samples (e.g. 3104 Evolutionary biology on the clinical papers of Life,
  3102 Bioinformatics on the clinical studies of Bioinformation, 3001 Agricultural biotechnology in
  Food Bioscience). The code is removed only from articles whose text gives it less than 10 %; in Journal
  of Voice, Music stays when it is the only label (papers on singing);
* records: labels that are specific groups (not "Other ..."), with at least 300 training examples, that
  get at most 3 % from the text while the text points to another group with at least 80 %;
* a record left without labels, or flagged in the second check, goes to train_excluded. Labels are only
  removed, never added or replaced; "Other ..." groups (codes ending in 99) are left as they are.
"""
import sys
from pathlib import Path

import pandas as pd

from common import DATA, REPORTS, save_json, say

FIXES = Path(__file__).resolve().parent / '06_label_fixes.csv'


def main():
    say('=== Крок 6: чистка міток у train ===\n')
    tr = pd.read_parquet(DATA / 'train.parquet')
    fx = pd.read_csv(FIXES, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    pos = {r: k for k, r in enumerate(tr.id)}
    missing = [r for r in fx.Record_ID if r not in pos]
    if missing:
        sys.exit(f'Записів із 06_label_fixes.csv, яких немає в train: {len(missing)} (напр. {missing[0]}). '
                 'Файл зроблено для іншого розбиття.')
    old = tr.labels.values
    differ = [r for r, o in zip(fx.Record_ID, fx.Old_labels) if old[pos[r]] != o]
    if differ:
        sys.exit(f'Записів, мітки яких уже не такі, як у 06_label_fixes.csv: {len(differ)} (напр. {differ[0]}).')

    labels = tr.labels.copy()
    excluded = set()
    for r, new, action in zip(fx.Record_ID, fx.New_labels, fx.Action):
        if action == 'excluded':
            excluded.add(r)
        else:
            labels.iat[pos[r]] = new
    clean = tr.assign(labels=labels)
    keep = ~clean.id.isin(excluded)
    out = clean[keep].reset_index(drop=True)
    out.to_parquet(DATA / 'train_clean.parquet', index=False)
    gone = tr[~keep].merge(fx[['Record_ID', 'Rule', 'Detail']], left_on='id', right_on='Record_ID').drop(columns='Record_ID')
    gone.to_parquet(DATA / 'train_excluded.parquet', index=False)

    count = lambda s: pd.Series([c for l in s for c in l.split(';') if c]).value_counts()
    before, after = count(tr.labels), count(out.labels)
    after = after.reindex(before.index, fill_value=0)
    change = ((after - before) / before).sort_values()
    jf = fx[fx.Rule.str.startswith('journal')].copy()
    jf['codes'] = jf.Detail.str.extract(r'^removed ([\d;]+):', expand=False).fillna('')
    by_journal = jf.groupby(jf.Source.str.lower()).agg(          # one journal may be spelled in several ways
        journal=('Source', lambda s: s.mode()[0]),
        codes=('codes', lambda s: ';'.join(sorted({c for x in s for c in x.split(';') if c}))),
        records=('Record_ID', 'size'), excluded=('Action', lambda a: int((a == 'excluded').sum())))
    by_journal = by_journal.sort_values('journal', key=lambda s: s.str.lower()).reset_index(drop=True)
    rep = {'train_in': len(tr), 'train_clean': len(out), 'excluded': len(gone),
           'labels_changed': int((fx.Action == 'code removed').sum()),
           'excluded_by_rule': gone.Rule.value_counts().to_dict(),
           'journals': by_journal.to_dict('records'),
           'groups_changed_most': [{'code': c, 'before': int(before[c]), 'after': int(after[c]),
                                    'change': round(float(v), 3)} for c, v in change.head(15).items()]}
    save_json(REPORTS / '06_clean_labels.json', rep)
    try:
        write_xlsx(REPORTS / '06_label_fixes.xlsx', tr, fx, rep, by_journal, before, after)
        xlsx = REPORTS / '06_label_fixes.xlsx'
    except PermissionError:
        say('(06_label_fixes.xlsx відкрито в Excel, тому його не оновлено.)')
        xlsx = None

    fmt = lambda x: f'{x:,}'.replace(',', ' ')
    say(f'train: {fmt(len(tr))} → train_clean: {fmt(len(out))}')
    say(f'  мітки виправлено (прибрано код) у {fmt(rep["labels_changed"])} записах; '
        f'коди, поставлені на весь журнал, — у {len(by_journal)} журналах')
    say(f'  вилучено {fmt(len(gone))} записів: ' + ', '.join(f'{RULES[k]} — {fmt(v)}' for k, v in rep['excluded_by_rule'].items()))
    say('Групи, що змінились найбільше: ' + '; '.join(
        f'{g["code"]} {g["before"]}→{g["after"]}' for g in rep['groups_changed_most'][:8]))
    say(f'\nЗбережено: {DATA / "train_clean.parquet"} і {DATA / "train_excluded.parquet"}')
    say(f'Звіт: {REPORTS / "06_clean_labels.json"}' + (f' і {xlsx} (усі зміни для перегляду)' if xlsx else ''))


RULES = {'journal code': 'код на весь журнал', 'record': 'мітки суперечать тексту',
         'journal code + record': 'обидві причини'}
ACTIONS = {'code removed': 'прибрано код', 'excluded': 'вилучено з train'}


def write_xlsx(path, tr, fx, rep, by_journal, before, after):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    from common import load_labels
    _, names, _ = load_labels()
    F = 'Arial'
    font, bold = Font(name=F, size=10), Font(name=F, size=10, bold=True)
    title, note = Font(name=F, size=13, bold=True), Font(name=F, size=9, italic=True, color='595959')
    head = PatternFill('solid', fgColor='D9E2F3')
    thin = Side(style='thin', color='BFBFBF')
    box = Border(top=thin, bottom=thin, left=thin, right=thin)
    wrap = Alignment(wrap_text=True, vertical='top')

    def table(ws, r0, heads, rows, fmts=None, widths=None):
        for k, h in enumerate(heads):
            c = ws.cell(r0, k + 1, h)
            c.font, c.fill, c.border, c.alignment = bold, head, box, wrap
        for i, row in enumerate(rows, 1):
            for k, v in enumerate(row):
                c = ws.cell(r0 + i, k + 1, v)
                c.font, c.border = font, box
                if fmts and fmts[k]:
                    c.number_format = fmts[k]
                if isinstance(v, str) and len(v) > 40:
                    c.alignment = wrap
        for k, w in enumerate(widths or []):
            ws.column_dimensions[get_column_letter(k + 1)].width = w
        return r0 + len(rows) + 1

    named = lambda s: '; '.join(f'{c} {names.get(c, "")}' for c in s.split(';') if c)
    wb = Workbook()
    ws = wb.active
    ws.title = 'Підсумок'
    ws['A1'], ws['A1'].font = 'Чистка міток у train (крок 6)', title
    ws['A2'] = ('Мітки лише прибрано, нових не додано; групи «Інше …» (коди на 99) не чіпали. '
                'Val і test не змінено, тому старі й нові моделі порівнюються на тих самих записах.')
    ws['A2'].font = note
    rows = [['Записів у train до чистки', rep['train_in']], ['Записів у train після чистки', rep['train_clean']],
            ['Прибрано код, запис лишився', rep['labels_changed']]]
    rows += [[f'Вилучено: {RULES[k]}', v] for k, v in rep['excluded_by_rule'].items()]
    end = table(ws, 4, ['Що', 'Записів'], rows, [None, '#,##0'], [62, 12, 16, 16])
    ws.cell(end + 1, 1, 'Коди, поставлені на весь журнал, яких тексти статей не підтримують').font = bold
    table(ws, end + 2, ['Журнал', 'Прибраний код', 'Записів змінено', 'З них вилучено (не лишилось міток)'],
          [[j.journal, named(j.codes), j.records, j.excluded] for j in by_journal.itertuples()],
          [None, None, '#,##0', '#,##0'])
    ws.column_dimensions['B'].width = 46
    ws.sheet_view.showGridLines = False

    ws = wb.create_sheet('Групи')
    ch = [(c, int(before[c]), int(after[c])) for c in before.index if after[c] != before[c]]
    ch.sort(key=lambda x: (x[2] - x[1]) / x[1])
    table(ws, 1, ['Код', 'Група', 'У train до', 'Після', 'Зміна'],
          [[c, names.get(c, ''), b, a, (a - b) / b] for c, b, a in ch], [None, None, '#,##0', '#,##0', '0.0%'],
          [7, 52, 11, 9, 9])
    ws.freeze_panes = 'A2'

    ws = wb.create_sheet('Усі зміни')
    title_of = dict(zip(tr.id, tr.title))
    rows = [[x.Record_ID, x.Source, title_of.get(x.Record_ID, ''), named(x.Old_labels), named(x.New_labels),
             ACTIONS.get(x.Action, x.Action), RULES.get(x.Rule, x.Rule), x.Detail] for x in fx.itertuples()]
    table(ws, 1, ['Record_ID', 'Журнал', 'Назва статті', 'Було', 'Стало', 'Дія', 'Причина', 'Деталі (англ.)'], rows,
          None, [15, 28, 60, 44, 44, 16, 22, 70])
    ws.freeze_panes, ws.auto_filter.ref = 'A2', ws.dimensions
    wb.save(path)


if __name__ == '__main__':
    main()
