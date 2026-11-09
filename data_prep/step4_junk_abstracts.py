#!/usr/bin/env python3
"""
ANZSRC dataset cleaning, step 4: service records that have an abstract, spam, and
abstracts that carry no content of their own.

    python step4_junk_abstracts.py <dataset_step3.csv> <title_only_step3.csv> <output_dir>

Main dataset:
  * removed: corrections / errata / retractions and retracted articles, obituaries and
    tributes, journal pages ("Cover Story"), editorial housekeeping notes (HOUSEKEEPING_IDS),
    spam (SPAM_IDS)
  * abstract judged to be junk -> Abstract = '', Has_usable_abstract = '0', record moved to
    the title-only file:
      adds_nothing  - after removing the title's words, stop words and words such as
                      data/file/supplementary/manuscript, at most 2 words are left
      citation      - the "abstract" is a bibliographic reference to the item itself
                      ("Miliaresis G., 2001. <title>. ASPRS Conf...", "Review of: ...",
                      "Cite this article as: ...")
      title_repeat  - the abstract repeats the title as "data / code / folder for <title>"
    junk abstract + a title with no content ("Untitled Item") -> removed
  * everything else, short abstracts included, stays unchanged.
Rows that stay are not modified; order is kept; moved rows are appended to the
title-only file.
"""
import json, os, sys
import pandas as pd
import re, unicodedata
def nt(x): return re.sub(r'[\W_]+',' ',unicodedata.normalize('NFKD',str(x)).casefold()).strip()
I = re.I
SERVICE = [
 ('correction', re.compile(
    r'^\W*(?:(?:author|authors|publisher|publisher s|editorial)\W*\s*)?(?:correction|corrigendum|corrigenda|erratum|errata|retraction|retracted(?: article)?|expression of concern|notice of retraction|withdrawal|withdrawn)'
    r'(?:\s+(?:note|notice|statement))?(?:\s*[:(\[<“"‘\'–—]|\s+-|\s+to\s*[:“"‘\'«<]|\s+of\s*:|\s+\w+ et al\b|\W*$)'
    r'|^\W*(?:erratum|corrigendum|corrigenda|retraction notice|retraction note)\s+to\b'
    r'|^\W*(?:erratum|errata|corrigendum|corrigenda)\s+(?:for|of|in)\b'
    r'|\bcorrigendum\W*$|^\W*editor\W*s note to\b', I)),
 ('journal_matter', re.compile(
    r'\b(?:front|back|end)\s*matter\b|^\W*cover(?:\s*\d|\s+(?:art|image|picture|story|page|photo|illustration|caption)\b|\s+and\s+(?:front|back)|\W*$)'
    r'|^\W*(?:table of )?contents\b|^\W*issue information\b|\bmasthead\b|^\W*(?:editorial\s+)?(?:editorial|advisory) board\b|\bed board\W*$'
    r'|^\W*(?:information|instructions|guidelines|notes?) for (?:authors|contributors|readers)\b|^\W*(?:author|subject|annual)?\s*index\W*$'
    r'|^\W*full issue\W*$|\baims\s*(?:/|and|&)?\s*scopes?\b', I)),
 ('announcement', re.compile(
    r'^\W*(?:announcement|announcing|award announcement|call for (?:papers|abstracts|submissions|nominations)|calendar of events|upcoming (?:events|meetings|conferences)|forthcoming (?:events|meetings))\b'
    r'|\bnewsletter\s*\(vol|^\W*newsletter\b|\bnews (?:for issue|and information page)\b|^\W*\d{4}\s+\w+\s+awards\W*$', I)),
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

import re, html, unicodedata
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
BOILER = set('''data dataset datasets file files folder supplementary supplemental supporting information material materials
contains contain containing collection thesis phd chapter figure figures fig table tables movie video appendix online article
paper manuscript raw used use using version please refer see available download code codes script scripts item items
related associated underlying minimal provided provides include includes including published publication journal entitled
titled study author authors pp vol volume issue proceedings proc conference conf workshop symposium press university
doi http https www org com html pdf xlsx csv docx zip txt accessed retrieved isbn issn eds ed editor editors abstract
title original present presents presented report reports record records description describes contained following
also one two three first second new main part parts number research project work works results result method methods
analysis analyses analysed analyzed file this these files'''.split())
WORD = re.compile(r"[^\W\d_]{3,}")
def clean(x):
    x = html.unescape(str(x)); x = re.sub(r'<[^>]+>', ' ', x); return re.sub(r'\s+', ' ', x).strip()
def words(x):
    x = unicodedata.normalize('NFKD', x).casefold()
    return [w for w in WORD.findall(x)]
def residual(title, abstract):
    t = {w[:6] for w in words(title)}
    res = [w for w in words(abstract) if w not in ENGLISH_STOP_WORDS and w not in BOILER and w[:6] not in t]
    return res
CITATION = re.compile(r'(?:\b(?:19|20)\d{2}\b.*\b(?:pp?\.|vol\.?|volume|journal|conf(?:erence)?|proc(?:eedings)?|in:|press|ed(?:s|itor)?\.?)\b)|'
                      r'(?:\b(?:pp?\.|vol\.)\s*\d)|(?:\b\d+\s*[-–]\s*\d+\b.*\b(?:19|20)\d{2}\b)', re.I)
NONALNUM = re.compile(r'[\W_]+')
def nalnum(x): return NONALNUM.sub('', unicodedata.normalize('NFKD', str(x)).casefold())
CITE_MARK = re.compile(r'\bin:\s|\bproceedings\b|\bproc\.|\bpp\.\s*\d|\bvol\.\s*\d|\bjournal\b|\bpress\b|\breview of:|\bbook reviewed:|'
                       r'\bplease cite\b|\bcite this article\b|\bproof copy\b|\bmonograph\b|\bpublished in\b', re.I)
YEAR_EARLY = re.compile(r'(?:,\s*|\(|\s)(?:19|20)\d{2}[a-z]?\s*[).,:]')
BIB_START = re.compile(r'^\W*(?:cite this article as|please cite as|book reviewed:|review of:|transmathematica archive\. please cite)', re.I)
CONTAINER = re.compile(r'\b(?:data|dataset|datasets|code|codes|script|scripts|folder|files?|materials?|supplement\w*|corpus|appendix|slides?|video|clip|poster|manual|chart|repository)\b', re.I)
NONALNUM = re.compile(r'[\W_]+')
def nalnum(x): return NONALNUM.sub('', unicodedata.normalize('NFKD', str(x)).casefold())
def title_in_abstract(t, a):
    nt_, na = nalnum(t), nalnum(a)
    if len(nt_) >= 15 and nt_ in na: return True
    tw = [w for w in words(t) if w not in ENGLISH_STOP_WORDS]
    if len(tw) >= 4:
        aw = set(words(a)); return sum(w in aw for w in tw) / len(tw) >= 0.85
    return False
def junk_reason(t, a):
    """'' = genuine; otherwise why the abstract adds nothing to the title"""
    res = residual(t, a); n = len(res)
    if n <= 2: return 'adds_nothing'
    tin = title_in_abstract(t, a)
    if BIB_START.search(a) and n <= 15: return 'citation'
    if tin and n <= 12 and (CITE_MARK.search(a) or YEAR_EARLY.search(a[:100])): return 'citation'
    if tin and n <= 4 and CONTAINER.search(a): return 'title_repeat'
    return ''

SPAM_IDS = {'mat_f5c5d08fecac6947a79c', 'mat_1fe0aa4eb74c6b7aa4cd', 'mat_89e8ddb378f3daa5ab5b'}
HOUSEKEEPING_IDS = {'orig_0080486', 'mat_031fde6b4455e42d4fb9', 'mat_ac16d0102f103d4ac89b'}
DELETE_SVC = {'correction', 'obituary', 'journal_matter', 'announcement', 'reviewers'}
REASON_UA = {
    'correction': 'Виправлення, errata, відкликання (зокрема відкликані статті)',
    'obituary': 'Некрологи, вшанування', 'journal_matter': 'Службові сторінки журналу',
    'announcement': 'Оголошення', 'reviewers': 'Подяка рецензентам',
    'housekeeping': 'Редакційні справи (вітання, зміни в журналі)', 'spam': 'Спам',
    'generic_title': 'Назва без змісту, а анотація — сміття',
    'adds_nothing': 'Анотація нічого не додає до назви',
    'citation': 'Анотація — бібліографічний опис самого запису',
    'title_repeat': 'Анотація лише повторює назву («дані / код до …»)',
}


def main(main_csv, title_csv, outdir):
    os.makedirs(outdir, exist_ok=True)
    rd = lambda f: pd.read_csv(f, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    d, t = rd(main_csv), rd(title_csv)
    cols = list(d.columns)
    svc = d.Title.map(classify)
    ab = d.Abstract.map(clean)
    nwords = ab.str.split().str.len()
    jr = pd.Series([junk_reason(ti, a) for ti, a in zip(d.Title, ab)], index=d.index)

    reason = pd.Series('', index=d.index)
    reason[svc.isin(DELETE_SVC)] = svc[svc.isin(DELETE_SVC)]
    reason[d.Record_ID.isin(HOUSEKEEPING_IDS)] = 'housekeeping'
    reason[d.Record_ID.isin(SPAM_IDS)] = 'spam'
    junk_generic = (jr != '') & (svc == 'generic_title') & (reason == '')
    reason[junk_generic] = 'generic_title'
    delete = reason != ''
    move = (jr != '') & ~delete

    moved = d[move].copy()
    moved['Abstract'] = ''
    moved['Has_usable_abstract'] = '0'
    d[~delete & ~move][cols].to_csv(os.path.join(outdir, 'dataset_step4.csv'), index=False, encoding='utf-8-sig')
    pd.concat([t, moved[cols]], ignore_index=True).to_csv(os.path.join(outdir, 'title_only.csv'), index=False,
                                                          encoding='utf-8-sig')
    log = pd.DataFrame({
        'action': ['видалено' if x else 'анотацію прибрано, запис перенесено в «лише назва»' for x in delete[delete | move]],
        'reason': [REASON_UA[r if r else j] for r, j in zip(reason[delete | move], jr[delete | move])],
        'Record_ID': d.Record_ID[delete | move], 'Record_origin': d.Record_origin[delete | move],
        'ANZSRC_4D': d.ANZSRC_4D[delete | move], 'Title': d.Title[delete | move],
        'Abstract_before': d.Abstract[delete | move].str[:400], 'words': nwords[delete | move]})
    log.to_pickle(os.path.join(outdir, 'step4_log.pkl'))
    kept = ~delete & ~move
    s = {'rows_in_main': len(d), 'rows_in_title_only': len(t), 'deleted': int(delete.sum()), 'moved': int(move.sum()),
         'main_out': int(kept.sum()), 'title_only_out': len(t) + int(move.sum()),
         'short_kept_lt40': int((kept & (nwords < 40)).sum()),
         'generic_title_kept': int((kept & (svc == 'generic_title')).sum())}
    json.dump(s, open(os.path.join(outdir, 'summary.json'), 'w'), indent=1)
    print(s)


if __name__ == '__main__':
    main(*sys.argv[1:4])
