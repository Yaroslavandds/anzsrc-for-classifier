#!/usr/bin/env python3
"""
ANZSRC dataset cleaning, step 6: FoR 2008 mappings checked against the official ABS
correspondence, and record language.

    python step6_for2008_language.py <dataset_step5.csv> <title_only_step5.csv> \
                                     <anzsrc_for2008_to_2020_ABS.csv> <output_dir>

ABS table: "1297.0 ANZSRC 2020 - Correspondence Tables", Table 2 "ANZSRC 2008 FoR -
ANZSRC 2020 FoR" (released 24 October 2025), file anzsrc2020_anzsrc2008_correspondences.xlsx
on abs.gov.au; columns code_2008, code_2020, partial ('p' = partial match).

1. Records whose ANZSRC holds FoR 2008 codes (fields 6-digit, groups 4-digit):
   allowed  = groups of FoR 2020 codes in ANZSRC + every FoR 2020 group the ABS table gives
              for the FoR 2008 codes (a 4-digit 2008 group = all its fields)
   required = groups of FoR 2020 codes in ANZSRC + groups of FoR 2008 codes that map to
              exactly one FoR 2020 group
   ANZSRC_4D := (ANZSRC_4D & allowed) | required
   -> groups the ABS table does not support are removed, unambiguous ones that were left
      out are added; groups chosen from ambiguous mappings stay.
2. Language (pycld2):
   * main file: English share of the abstract < 30 % -> record moved to non_english.csv
     (bilingual abstracts with an English part of 30 % or more stay);
     abstract that is mostly '?' (lost encoding) -> abstract removed, record moved to
     the title-only file
   * title-only file: title in a non-Latin script, or detected as non-English, without
     English function words and with < 70 % of its words found in the English abstracts
     of the main file -> moved to non_english.csv
Rows are otherwise unchanged; order kept; same 26 columns in every file.
"""
import html, json, os, re, sys, unicodedata, collections
import pandas as pd
import pycld2 as cld2

MIN_EN_SHARE = 30
CODE = re.compile(r'(?<!\d)(\d{2}|\d{4}|\d{6})(?!\d)')
TAG = re.compile(r'<[^>]+>')
WORD = re.compile(r'[a-z]{3,}')
EN_WORDS = set('the of and for with from by on an is are this that its their between among into using based '
               'toward towards study analysis review effect effects impact'.split())


def load_abs(path):
    t = pd.read_csv(path, dtype=str, keep_default_na=False)
    f2g = collections.defaultdict(set)
    for a, b in zip(t.code_2008, t.code_2020):
        f2g[a].add(b[:4])
    g2g = collections.defaultdict(set)
    for a, gs in f2g.items():
        g2g[a[:4]] |= gs
    return f2g, g2g


def fix_2008(anzsrc, l4d, f2g, g2g):
    codes = CODE.findall(anzsrc)
    c08 = [c for c in codes if len(c) in (4, 6) and 1 <= int(c[:2]) <= 22]
    if not c08:
        return l4d, '', '', ''
    g20 = {c[:4] for c in codes if len(c) in (4, 6) and int(c[:2]) >= 30}
    allowed, required = set(g20), set(g20)
    for c in c08:
        g = f2g.get(c, set()) if len(c) == 6 else g2g.get(c, set())
        allowed |= g
        if len(g) == 1:
            required |= g
    old = {c for c in l4d.split(';') if c}
    new = (old & allowed) | required
    return ';'.join(sorted(new)), ';'.join(sorted(old - new)), ';'.join(sorted(new - old)), 'checked'


def clean(x):
    return re.sub(r'\s+', ' ', TAG.sub(' ', html.unescape(str(x)))).strip()


def lang(text):
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    if not text:
        return '', 0, ''
    try:
        _, _, det = cld2.detect(text, bestEffort=True)
    except Exception:
        return '', 0, ''
    en = sum(p for _, c, p, _ in det if c == 'en')
    top = det[0][1] if det[0][1] != 'un' else ''
    return top, en, ';'.join(f'{c}:{p}' for _, c, p, _ in det if c != 'un')


def latin_share(s):
    letters = [ch for ch in s if ch.isalpha()]
    return sum(ch.isascii() or unicodedata.name(ch, '').startswith('LATIN') for ch in letters) / max(1, len(letters))


def main(main_csv, title_csv, abs_csv, outdir):
    os.makedirs(outdir, exist_ok=True)
    rd = lambda f: pd.read_csv(f, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    m, t = rd(main_csv), rd(title_csv)
    cols = list(m.columns)
    f2g, g2g = load_abs(abs_csv)

    fixes = []
    for name, df in (('основний', m), ('лише назва', t)):
        res = [fix_2008(a, l, f2g, g2g) for a, l in zip(df.ANZSRC, df.ANZSRC_4D)]
        new = [r[0] for r in res]
        for i, (r, old) in enumerate(zip(res, df.ANZSRC_4D)):
            if r[1] or r[2]:
                fixes.append({'file': name, 'Record_ID': df.Record_ID.iat[i], 'Record_origin': df.Record_origin.iat[i],
                              'ANZSRC': df.ANZSRC.iat[i], 'ANZSRC_4D_before': old, 'ANZSRC_4D_after': r[0],
                              'removed': r[1], 'added': r[2], 'Label_origin': df.Label_origin.iat[i], 'Title': df.Title.iat[i]})
        df['ANZSRC_4D'] = new
    n2008 = int(m.ANZSRC.map(lambda a: any(len(c) in (4, 6) and 1 <= int(c[:2]) <= 22 for c in CODE.findall(a))).sum())

    # language of the main file (abstract)
    ab = m.Abstract.map(clean)
    det = [lang(a) for a in ab]
    m_top = pd.Series([d[0] for d in det], index=m.index)
    m_en = pd.Series([d[1] for d in det], index=m.index)
    m_mix = pd.Series([d[2] for d in det], index=m.index)
    garbled = ab.map(lambda s: s.count('?') / max(1, len(s.replace(' ', '')))) > 0.3
    non_en_main = (m_en < MIN_EN_SHARE) & ~garbled
    # language of the title-only file (title)
    ti = t.Title.map(clean)
    tdet = [lang(x) for x in ti]
    t_top = pd.Series([d[0] for d in tdet], index=t.index)
    t_mix = pd.Series([d[2] for d in tdet], index=t.index)
    en_words = ti.map(lambda s: sum(w in EN_WORDS for w in re.findall(r'[a-z]+', s.lower())))
    non_latin = ti.map(latin_share) < 0.5
    # short titles fool CLD2 ("ORAL penicillin.", "Conrad's gout."): a title whose words are
    # mostly words used in the English abstracts of the main file counts as English
    df_cnt = collections.Counter()
    for a in ab[m_en >= MIN_EN_SHARE]:
        df_cnt.update(set(WORD.findall(a.lower())))
    vocab = {w for w, n in df_cnt.items() if n >= 5}
    def vocab_share(s):
        ws = WORD.findall(s.lower())
        return sum(w in vocab for w in ws) / len(ws) if ws else 0
    t_vocab = ti.map(vocab_share)
    non_en_title = non_latin | ((t_top != 'en') & (en_words == 0) & (t_vocab < 0.7))

    moved_junk = m[garbled].copy()
    moved_junk['Abstract'] = ''
    moved_junk['Has_usable_abstract'] = '0'
    main_out = m[~non_en_main & ~garbled]
    title_out = pd.concat([t[~non_en_title], moved_junk], ignore_index=True)
    non_en = pd.concat([m[non_en_main], t[non_en_title]], ignore_index=True)
    main_out[cols].to_csv(os.path.join(outdir, 'dataset_step6.csv'), index=False, encoding='utf-8-sig')
    title_out[cols].to_csv(os.path.join(outdir, 'title_only.csv'), index=False, encoding='utf-8-sig')
    non_en[cols].to_csv(os.path.join(outdir, 'non_english.csv'), index=False, encoding='utf-8-sig')

    lang_log = pd.concat([
        pd.DataFrame({'from_file': 'основний', 'to_file': 'non_english', 'Record_ID': m.Record_ID[non_en_main],
                      'language': m_top[non_en_main], 'languages_detected': m_mix[non_en_main],
                      'english_share': m_en[non_en_main], 'ANZSRC_4D': m.ANZSRC_4D[non_en_main],
                      'Title': m.Title[non_en_main], 'text_start': ab[non_en_main].str[:200]}),
        pd.DataFrame({'from_file': 'лише назва', 'to_file': 'non_english', 'Record_ID': t.Record_ID[non_en_title],
                      'language': t_top[non_en_title], 'languages_detected': t_mix[non_en_title],
                      'english_share': 0, 'ANZSRC_4D': t.ANZSRC_4D[non_en_title], 'Title': t.Title[non_en_title],
                      'text_start': ''}),
        pd.DataFrame({'from_file': 'основний', 'to_file': 'лише назва (анотація — знаки «?»)',
                      'Record_ID': m.Record_ID[garbled], 'language': '', 'languages_detected': '', 'english_share': 0,
                      'ANZSRC_4D': m.ANZSRC_4D[garbled], 'Title': m.Title[garbled], 'text_start': ab[garbled].str[:200]})])
    bilingual = pd.DataFrame({'Record_ID': m.Record_ID, 'language': m_top, 'languages_detected': m_mix,
                              'english_share': m_en, 'Title': m.Title, 'text_start': ab.str[:200]})
    bilingual = bilingual[(m_top != 'en') & ~non_en_main & ~garbled]
    pd.DataFrame(fixes).to_pickle(os.path.join(outdir, 'for2008_fixes.pkl'))
    lang_log.to_pickle(os.path.join(outdir, 'language_moves.pkl'))
    bilingual.to_pickle(os.path.join(outdir, 'bilingual_kept.pkl'))
    s = {'main_in': len(m), 'title_in': len(t), 'main_with_for2008': n2008,
         'for2008_fixed_main': sum(f['file'] == 'основний' for f in fixes),
         'for2008_fixed_title': sum(f['file'] == 'лише назва' for f in fixes),
         'main_to_non_english': int(non_en_main.sum()), 'title_to_non_english': int(non_en_title.sum()),
         'main_garbled_to_title': int(garbled.sum()), 'bilingual_kept': len(bilingual),
         'main_out': len(main_out), 'title_out': len(title_out), 'non_english_out': len(non_en)}
    json.dump(s, open(os.path.join(outdir, 'summary.json'), 'w'), indent=1)
    print(s)


if __name__ == '__main__':
    main(*sys.argv[1:5])
