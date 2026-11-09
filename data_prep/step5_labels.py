#!/usr/bin/env python3
"""
ANZSRC dataset cleaning, step 5: labels, plus records with no subject information.

    python step5_labels.py <dataset_step4.csv> <title_only_step4.csv> <output_dir>

1. Removed (main file):
   * records with a generic title ("Preface", "Introduction", "Book reviews" ...) - prefaces,
     editorials and book-review sections rather than research
   * corrections / calls for papers caught by the extended patterns
   * NO_SUBJECT_IDS: file descriptions whose title and abstract say nothing about the topic
2. ANZSRC_4D vs the source field ANZSRC (checked for every record):
   * FoR 2008 codes in ANZSRC: ANZSRC_4D holds their FoR 2020 groups (mapping done when the
     data were collected) - expected, left as is
   * original records matched to a university repository: ANZSRC_4D also holds the
     repository's FoR codes, which ANZSRC does not list - real labels, left as is
   * groups of FoR 2020 codes that ANZSRC lists but ANZSRC_4D lacks - added
3. Records with more than MAX_LABELS groups are removed (indiscriminate self-tagging).
The same rules are applied to the title-only file. ANZSRC itself is never changed.
"""
import json, os, re, sys
import pandas as pd
import re, unicodedata
def nt(x): return re.sub(r'[\W_]+',' ',unicodedata.normalize('NFKD',str(x)).casefold()).strip()
I = re.I
SERVICE = [
 ('correction', re.compile(
    r'^\W*(?:(?:author|authors|publisher|publisher s|editorial)\W*\s*)?(?:correction|corrigendum|corrigenda|erratum|errata|retraction|retracted(?: article)?|expression of concern|notice of retraction|withdrawal|withdrawn)'
    r'(?:\s+(?:note|notice|statement))?(?:\s*[:(\[<“"‘\'–—]|\s+-|\s+to\s*[:“"‘\'«<]|\s+of\s*:|\s+\w+ et al\b|\W*$)'
    r'|^\W*(?:erratum|corrigendum|corrigenda|retraction notice|retraction note)\s+to\b'
    r'|^\W*(?:erratum|errata|corrigendum|corrigenda)\b'
    r'|^\W*(?:erratum|errata|corrigendum|corrigenda)\s+(?:for|of|in)\b'
    r'|\bcorrigendum\W*$|^\W*editor\W*s note to\b', I)),
 ('journal_matter', re.compile(
    r'\b(?:front|back|end)\s*matter\b|^\W*cover(?:\s*\d|\s+(?:art|image|picture|story|page|photo|illustration|caption)\b|\s+and\s+(?:front|back)|\W*$)'
    r'|^\W*(?:table of )?contents\b|^\W*issue information\b|\bmasthead\b|^\W*(?:editorial\s+)?(?:editorial|advisory) board\b|\bed board\W*$'
    r'|^\W*(?:information|instructions|guidelines|notes?) for (?:authors|contributors|readers)\b|^\W*(?:author|subject|annual)?\s*index\W*$'
    r'|^\W*full issue\W*$|\baims\s*(?:/|and|&)?\s*scopes?\b', I)),
 ('announcement', re.compile(
    r'^\W*(?:announcement|announcing|award announcement|call for (?:papers|abstracts|submissions|nominations)|calendar of events|upcoming (?:events|meetings|conferences)|forthcoming (?:events|meetings))\b'
    r'|\bcall for (?:papers|abstracts|submissions)\b|\bnewsletter\s*\(vol|^\W*newsletter\b|\bnews (?:for issue|and information page)\b|^\W*\d{4}\s+\w+\s+awards\W*$', I)),
 ('obituary', re.compile(
    r'\bin memoriam\b|^\W*(?:obituary|obituaries|in memory of)\b|^\W*(?:a )?tribute(?: to\b|\s*:)'
    , I)),
 ('obituary', re.compile(r'^\W*(?:[A-ZÀ-ÖØ-Þ][\w.\'’-]*\s+){1,3}[A-ZÀ-ÖØ-Þ][\w.\'’-]*\s*\(\s*\d{4}\s*[-–—]\s*\d{4}\s*\)\W*$')),
 ('reviewers', re.compile(r'^\W*(?:acknowledg\w* (?:of|to|our)?\s*(?:the\s+)?reviewers|thank(?:s| you) to (?:our )?reviewers|(?:list of )?reviewers?(?: list| \d{4})?\W*$|reviewer acknowledg\w*)', I)),
]
GENERIC = {nt(x) for x in '''editorial|guest editorial|editors note|editor s note|editors notes|from the editor|from the editors|from the editor in chief|
letter from the editor|editor s introduction|editorial introduction|introduction|preface|foreword|book review|book reviews|reviews|abstracts|abstract|
reply|response|commentary|letter to the editor|letters to the editor|in this issue|highlights|news|notes|correspondence|announcements|editorial comment|
editorial note|note from the editor|untitled|untitled item|title|no title|n a|editorial highlights|editor s message|editors message|book reviews editorial|
conversations editorial|letter to our readers|readers information|response to letter to the editor|authors response to the letter to the editor|
thesis|remarkable image|new resources|vraag antwoord|crossword puzzle|society page|cme accreditation page|subscription information|author guidelines|
aaap sponsors|sponsors|acknowledgements|acknowledgments'''.replace('\n','').split('|')}
def classify(title):
    for name, rx in SERVICE:
        if rx.search(title): return name
    if nt(title) in GENERIC: return 'generic_title'
    return ''

MAX_LABELS = 6
NO_SUBJECT_IDS = {'mat_f5c215a985fb8a18bd95', 'mat_bfacb182f7f7cd33b654', 'mat_e8d75f5e58f1ac4639b0',
                  'mat_6a7047863f93e8aa2edd', 'mat_09bc35aff2d800a3ee49', 'mat_8592f9c6c83afdc2a108'}
CODE = re.compile(r'(?<!\d)(\d{2}|\d{4}|\d{6})(?!\d)')
REASON_UA = {'generic_title': 'Загальна назва («Preface», «Introduction», «Book reviews» …)',
             'correction': 'Виправлення / errata', 'announcement': 'Оголошення (call for papers)',
             'obituary': 'Некрологи', 'journal_matter': 'Службові сторінки журналу', 'reviewers': 'Подяка рецензентам',
             'no_subject': 'Ні назва, ні анотація нічого не кажуть про тему',
             'too_many_labels': f'Більше {MAX_LABELS} міток (мітки поставлено «все підряд»)'}
FIX_UA = {'missing_added': 'Додано групи кодів FoR 2020, які є в ANZSRC'}


def groups_2020(anzsrc):
    return {c[:4] for c in CODE.findall(anzsrc) if len(c) in (4, 6) and int(c[:2]) >= 30}


def has_2008(anzsrc):
    return any(len(c) in (4, 6) and 1 <= int(c[:2]) <= 22 for c in CODE.findall(anzsrc))


def fix_labels(anzsrc, l4d):
    """only adds groups of FoR 2020 codes that ANZSRC lists and ANZSRC_4D lacks"""
    old = {c for c in re.split(r'[;,\s]+', l4d) if c}
    new = old | groups_2020(anzsrc)
    return ';'.join(sorted(new)), ('missing_added' if new != old else '')


def process(d, main=True):
    reason = pd.Series('', index=d.index)
    svc = d.Title.map(classify)
    reason[svc != ''] = svc[svc != '']
    reason[d.Record_ID.isin(NO_SUBJECT_IDS)] = 'no_subject'
    fixed = [fix_labels(a, l) for a, l in zip(d.ANZSRC, d.ANZSRC_4D)]
    new4d = pd.Series([f[0] for f in fixed], index=d.index)
    kind = pd.Series([f[1] for f in fixed], index=d.index)
    nlab = new4d.str.split(';').str.len()
    reason[(reason == '') & (nlab > MAX_LABELS)] = 'too_many_labels'
    out = d.copy()
    out['ANZSRC_4D'] = new4d
    keep = reason == ''
    dele = pd.DataFrame({'file': 'основний' if main else 'лише назва', 'reason': reason[~keep].map(REASON_UA),
                         'Record_ID': d.Record_ID[~keep], 'Record_origin': d.Record_origin[~keep],
                         'ANZSRC_4D': d.ANZSRC_4D[~keep], 'n_labels': nlab[~keep], 'Title': d.Title[~keep]})
    ch = keep & (kind != '')
    fixes = pd.DataFrame({'file': 'основний' if main else 'лише назва', 'fix': kind[ch].map(FIX_UA),
                          'Record_ID': d.Record_ID[ch], 'Record_origin': d.Record_origin[ch], 'ANZSRC': d.ANZSRC[ch],
                          'ANZSRC_4D_before': d.ANZSRC_4D[ch], 'ANZSRC_4D_after': new4d[ch], 'Title': d.Title[ch]})
    return out[keep], dele, fixes


def main(main_csv, title_csv, outdir):
    os.makedirs(outdir, exist_ok=True)
    rd = lambda f: pd.read_csv(f, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    d, t = rd(main_csv), rd(title_csv)
    cols = list(d.columns)
    m_out, m_del, m_fix = process(d, True)
    t_out, t_del, t_fix = process(t, False)
    m_out[cols].to_csv(os.path.join(outdir, 'dataset_step5.csv'), index=False, encoding='utf-8-sig')
    t_out[cols].to_csv(os.path.join(outdir, 'title_only.csv'), index=False, encoding='utf-8-sig')
    pd.concat([m_del, t_del]).to_pickle(os.path.join(outdir, 'deleted.pkl'))
    pd.concat([m_fix, t_fix]).to_pickle(os.path.join(outdir, 'fixed.pkl'))
    s = {'main_in': len(d), 'main_out': len(m_out), 'main_deleted': len(m_del), 'main_fixed': len(m_fix),
         'title_in': len(t), 'title_out': len(t_out), 'title_deleted': len(t_del), 'title_fixed': len(t_fix),
         'main_with_2008': int(d.ANZSRC.map(has_2008).sum())}
    json.dump(s, open(os.path.join(outdir, 'summary.json'), 'w'), indent=1)
    print(s)


if __name__ == '__main__':
    main(*sys.argv[1:4])
