#!/usr/bin/env python3
"""
ANZSRC dataset cleaning, step 3: records without a usable abstract.

    python step3_title_only.py <step2.csv> <output_dir>

  * records with Has_usable_abstract = 1        -> dataset_step3.csv   (main dataset)
  * records with Has_usable_abstract != 1:
      - service records (corrections, errata, retractions, journal front/back matter,
        announcements, obituaries/tributes, bare generic titles such as "Editorial",
        "Introduction", "From the Editor-in-Chief")                 -> removed, listed in the log
      - everything else (title-only research records)               -> title_only.csv
Rows are written unchanged, in their original order. Patterns were checked by hand
against every matching title in the dataset.
"""
import json, os, sys
import pandas as pd
import re, unicodedata
def nt(x): return re.sub(r'[\W_]+',' ',unicodedata.normalize('NFKD',str(x)).casefold()).strip()
I = re.I
SERVICE = [
 ('correction', re.compile(
    r'^\W*(?:(?:author|authors|publisher|publisher s|editorial)\W*\s*)?(?:correction|corrigendum|corrigenda|erratum|errata|retraction|retracted(?: article)?|expression of concern|notice of retraction|withdrawal|withdrawn)'
    r'(?:\s+(?:note|notice|statement))?(?:\s*[:\-–—(\[<“"‘\']|\s+to\s*[:“"‘\'«<]|\s+of\s*:|\s+\w+ et al\b|\W*$)'
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

CATEGORY_UA = {'correction': 'Виправлення, errata, відкликання', 'journal_matter': 'Службові сторінки журналу',
               'announcement': 'Оголошення, новини, нагороди', 'obituary': 'Некрологи, вшанування',
               'reviewers': 'Подяка рецензентам', 'generic_title': 'Назва без змісту («Editorial», «Introduction» …)'}


def main(inp, outdir):
    os.makedirs(outdir, exist_ok=True)
    d = pd.read_csv(inp, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    usable = d.Has_usable_abstract.str.strip().eq('1')
    svc = d.Title.map(classify)
    service = ~usable & svc.ne('')
    title_only = ~usable & svc.eq('')
    d[usable].to_csv(os.path.join(outdir, 'dataset_step3.csv'), index=False, encoding='utf-8-sig')
    d[title_only].to_csv(os.path.join(outdir, 'title_only.csv'), index=False, encoding='utf-8-sig')
    removed = d[service].assign(category=svc[service].map(CATEGORY_UA))
    removed.to_pickle(os.path.join(outdir, 'removed_service.pkl'))
    s = {'rows_in': len(d), 'main': int(usable.sum()), 'title_only': int(title_only.sum()),
         'removed_service': int(service.sum()),
         'title_only_with_unusable_text': int((title_only & d.Abstract.str.strip().ne('')).sum()),
         'service_with_abstract_left_in_main': int((usable & svc.ne('')).sum())}
    json.dump(s, open(os.path.join(outdir, 'summary.json'), 'w'), indent=1)
    print(s)


if __name__ == '__main__':
    main(*sys.argv[1:3])
