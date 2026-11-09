#!/usr/bin/env python3
"""
ANZSRC dataset cleaning, step 2: records that share one identical abstract
under different titles (left over from step 1).

    python step2_shared_abstracts.py <step1.csv> <output_dir>

Every group of records with an identical abstract (>= 100 normalised chars) was
reviewed by hand and given one action (DECISIONS below, keyed by an md5 prefix of
the normalised abstract):

  BLANK         the abstract is boilerplate or belongs to another work, the titles are
                real and informative -> abstract removed, title kept
                (Abstract = '', Has_usable_abstract = '0')
  DELETE        not research output, or boilerplate with titles that say nothing
                -> records removed
  KEEP1         parts, files, versions, translations or issues of one item that share
                one description -> one record kept (the title that best matches the
                abstract, then the step-1 priority), the others removed
  BLANK_OTHERS  different works; the abstract belongs to one of them -> that record
                keeps it (best title/abstract match, or KEEPER override), the others
                keep only their title

Records outside the groups that carry the same boilerplate are treated the same
way (PATTERNS): UIC library newsletter items and Pfizer congress materials are
removed, artist-biography "abstracts" are removed, Cairn publisher notices blanked.
All other records are written unchanged, in their original order.
"""
import hashlib, html, json, os, re, sys, unicodedata
import pandas as pd

TAG = re.compile(r'<[^>]+>')
NONWORD = re.compile(r'[\W_]+')
WORD = re.compile(r'[^\W\d_]{3,}')
STOP = set('the and for from with that this into over under about between using based study data paper '
           'analysis new its their our are was were has have via des les une pour dans del las los por con '
           'para una que und der die das'.split())
ORIGIN_RANK = {'original': 0, 'original_updated': 0, 'new': 1, 'new_material': 2}


def norm_text(x):
    x = TAG.sub(' ', html.unescape(str(x)))
    return NONWORD.sub('', unicodedata.normalize('NFKD', x).casefold())


def key(a):
    return hashlib.md5(norm_text(a).encode()).hexdigest()[:10]


def stems(x):
    x = unicodedata.normalize('NFKD', html.unescape(str(x))).casefold()
    return {w[:5] for w in WORD.findall(x) if w not in STOP}


B, D, K, O = 'BLANK', 'DELETE', 'KEEP1', 'BLANK_OTHERS'
TYPES = {
    'boilerplate': 'Шаблонна анотація, змістовні назви',
    'non_research': 'Не наукові записи',
    'empty_titles': 'Шаблонна анотація, назви без змісту',
    'series': 'Частини, файли, версії чи переклади одного запису',
    'wrong_abstract': 'Чужа анотація',
    'belongs_to_one': 'Різні роботи, анотація належить одній з них',
}
# md5 prefix of normalised abstract -> (action, type, example)
DECISIONS = {
    'a0e6f12e3d': (B, 'boilerplate', 'APCWE-IX wind engineering papers: peer-review notice'),
    'ca05f6c292': (B, 'boilerplate', 'Monash University Law Review notice'),
    '3e12cf3beb': (B, 'boilerplate', 'Monash University Law Review notice (variant)'),
    'c47b6cf47d': (B, 'boilerplate', 'journal description: Biosciences Biotechnology Research Asia'),
    'f3f277cc2f': (B, 'boilerplate', 'journal description: J Colleges of Medicine of South Africa'),
    '8fae1a616e': (B, 'boilerplate', 'book description: Macquarie Atlas of Indigenous Australia chapters'),
    'fdb82d63b7': (B, 'boilerplate', 'book description: Cinemeducation chapters'),
    '6242ff74c0': (B, 'boilerplate', 'ESIP Summer Meeting session notice'),
    '95a0bedc05': (B, 'boilerplate', 'ESIP Winter Meeting notice'),
    '1eeefbaf62': (B, 'boilerplate', 'FORCE meeting presentation notice'),
    'c0a9b5c3a0': (B, 'boilerplate', 'conference citation instead of abstract'),
    '1c60b712ff': (B, 'boilerplate', 'Cairn publisher notice'),
    'baecc65360': (B, 'wrong_abstract', 'French art reviews with an abstract on intellectual property'),
    '004380faf3': (B, 'wrong_abstract', 'French art reviews with a review of another book'),
    '900ea5e354': (B, 'wrong_abstract', 'French art reviews with a review of another book'),
    '57da211809': (D, 'non_research', 'UIC Library of the Health Sciences newsletter'),
    'c7a97c1bfd': (D, 'non_research', 'Pfizer congress materials, disclaimer'),
    '50346e2f4f': (D, 'non_research', 'Pfizer congress materials, disclaimer'),
    'a24b11a12a': (D, 'non_research', 'Pfizer congress materials, disclaimer'),
    '8a559d4884': (D, 'non_research', 'Pfizer congress materials, disclaimer'),
    '427ff3cdef': (D, 'non_research', 'Pfizer congress materials, disclaimer'),
    '6017ef6363': (D, 'non_research', 'Pfizer congress materials, disclaimer'),
    'a4a8e81015': (D, 'non_research', 'Pfizer congress materials, disclaimer'),
    'b74cfa75e8': (D, 'non_research', 'Pfizer congress materials, disclaimer'),
    'af8e1fc6ae': (D, 'non_research', 'Pfizer congress materials, disclaimer'),
    '75b5abc57a': (D, 'non_research', 'private student portfolio'),
    'c5e167aa9b': (D, 'empty_titles', 'Bloomsbury Project website archive folders'),
    '84e4ad0064': (D, 'empty_titles', 'MacroSheds access notice, "Network: x, Domain: y"'),
    '7046918b70': (D, 'empty_titles', 'artist biography as abstract, titles of images'),
    'd17ce8580b': (D, 'empty_titles', '"Supplementary file 1/2", data-availability note'),
    'ab703d5e32': (O, 'belongs_to_one', 'storytelling abstract on 4 articles'),
    '384256a88f': (O, 'belongs_to_one', 'monetary policy commentary'),
    '2159e14c05': (O, 'belongs_to_one', 'Indigenous VET presentations'),
    '1c508cef5b': (O, 'belongs_to_one', 'performance measurement, telecom'),
    '7cbb15ced8': (O, 'belongs_to_one', 'Indonesian: network training vs bank queue simulation'),
    '7bef3db587': (O, 'belongs_to_one', 'rhamnolipids, two papers'),
    '7bc4adb6c4': (O, 'belongs_to_one', 'Aboriginal literature and policing'),
    '6559abeb35': (O, 'belongs_to_one', 'authenticity in experiences, two papers'),
    '59a16354d4': (O, 'belongs_to_one', 'Indigenous language learning, two papers'),
    '4e9c1b550f': (O, 'belongs_to_one', 'eco-enzyme vs patchouli shade'),
    '1faef84ad8': (O, 'belongs_to_one', 'Caution Bay chapters'),
    '318d48bfe5': (O, 'belongs_to_one', 'nerve stimulation, two case reports'),
    'c01a7b9b05': (O, 'belongs_to_one', 'biophilic urbanism, two reports'),
    'b9cd36d2a1': (O, 'belongs_to_one', 'memory systems, two papers'),
    'b2debbac96': (O, 'belongs_to_one', 'online creativity, two items'),
    'c989c6f78b': (O, 'belongs_to_one', 'photocatalytic disinfection chapters'),
    'cba98a82c7': (O, 'belongs_to_one', 'monetary policy, two essays'),
    'e218c0b1e0': (O, 'belongs_to_one', 'legal environment for selection, two papers'),
}
SERIES = '''a7bbc05216 a1211f7a17 9ee5940b4c 6cff673807 36f8eecd6d 08ce06ea46 02361bb41a 3f3d7cf71b 11abc5e654
b82f10e3b6 fc1f2f8a12 59ddd7e31e 3608c0c0a2 b6b1cde73c ad844c75d6 0716e71b12 860846cc02 5aef75c6a6 f583ce425f
8dc5f79819 8de51c053d f783951f2b eeb86e740a 222ee3a584 0b265515b0 7a1ad8d948 45bf202825 421a8459b5 465bb92689
1e3cd862f5 0e13efa026 2619376668 7e391a77fb 734b1953e4 71ab41b01a 64846a52a5 67bcfcc33c 64ab7ccc98 64bc7ad1dd
6268d5469b 5aadb67539 5d1b1193c1 49d36968fe 4a67a823dd 51994e5e4b 509989c7bc 51468bc1a4 57a1373c62 30738dad36
3afbcadad9 908cc913d9 901aac9f35 84fc405187 93ab9d5a70 c05c572cfc b7cde6e32e 9a5cee3826 812d485c04 d4041f68ae
d4aca50206 d45e70ec1c e201ec0c99 e1542b8841 d8a33949d6 d82319a419 d765867e2b e57513c221 e405e5b639 f1ba63f902
f0450a0835 e70ee19e72 e5d187aa53 f415d9b56b f9e984201b fed22a5d2a'''.split()
for h in SERIES:
    DECISIONS[h] = (K, 'series', '')
# checked by hand: whose abstract it is (normalised title prefix)
KEEPER_TITLE = {'ab703d5e32': 'narrativeembodimentandhealth',     # storytelling, embodiment, health
                '384256a88f': 'commentarytheroleofjudgment'}       # "When ... invited me to discuss Friedman's paper"
# same boilerplate outside the groups
PATTERNS = [
    (r'originally published in the UIC LHS Newsletter', D, 'non_research'),
    (r'^Congress materials may include information for investigational products', D, 'non_research'),
    (r'Philip Alexander, MD', D, 'empty_titles'),
    (r'Découvrir le sommaire de ce numéro, suivre la revue par email', B, 'boilerplate'),
]
ACTION_UA = {'deleted': 'видалено', 'blanked': 'анотацію прибрано, назва лишилась',
             'kept': 'лишився без змін', 'kept_with_abstract': 'лишився з анотацією'}


def main(inp, outdir):
    os.makedirs(outdir, exist_ok=True)
    d = pd.read_csv(inp, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    cols = list(d.columns)
    d['_a'] = d.Abstract.map(norm_text)
    d['_t'] = d.Title.map(norm_text)
    d['_h'] = d._a.map(lambda a: hashlib.md5(a.encode()).hexdigest()[:10])
    d['_match'] = [len(stems(t) & stems(a)) / max(1, len(stems(t))) for t, a in zip(d.Title, d.Abstract)]
    d['_rank'] = list(zip(-d._match, d.Has_usable_abstract.ne('1').astype(int), d.Record_origin.map(ORIGIN_RANK).fillna(3),
                          -d._a.str.len(), d.DOI.eq('').astype(int), d.Record_ID))
    shared = d[(d._a.str.len() >= 100) & d.duplicated('_a', keep=False)]
    groups = sorted(shared._h.unique())
    missing = [h for h in groups if h not in DECISIONS]
    unused = [h for h in DECISIONS if h not in set(groups)]
    assert not missing and not unused, (missing, unused)

    log = []  # (index, group, type, action)
    drop, blank = set(), set()
    for h in groups:
        act, typ, _ = DECISIONS[h]
        g = shared[shared._h == h]
        g = g.loc[sorted(g.index, key=lambda i: d.at[i, '_rank'])]
        if act == B:
            blank |= set(g.index); log += [(i, h, typ, 'blanked') for i in g.index]
        elif act == D:
            drop |= set(g.index); log += [(i, h, typ, 'deleted') for i in g.index]
        else:
            first = g.index[0]
            if h in KEEPER_TITLE:
                first = g.index[g._t.str.startswith(KEEPER_TITLE[h])][0]
            rest = [i for i in g.index if i != first]
            if act == K:
                drop |= set(rest)
                log += [(first, h, typ, 'kept')] + [(i, h, typ, 'deleted') for i in rest]
            else:
                blank |= set(rest)
                log += [(first, h, typ, 'kept_with_abstract')] + [(i, h, typ, 'blanked') for i in rest]
    done = {i for i, *_ in log}
    for pat, act, typ in PATTERNS:
        m = d.Abstract.str.contains(pat, regex=True) & ~d.index.isin(done)
        for i in d.index[m]:
            (drop if act == D else blank).add(i)
            log.append((i, 'pattern', typ, 'deleted' if act == D else 'blanked'))
            done.add(i)

    out = d.copy()
    out.loc[list(blank), 'Abstract'] = ''
    out.loc[list(blank), 'Has_usable_abstract'] = '0'
    out = out.drop(index=list(drop))
    out[cols].to_csv(os.path.join(outdir, 'dataset_step2.csv'), index=False, encoding='utf-8-sig')

    L = pd.DataFrame(log, columns=['i', 'group', 'type', 'action'])
    L['size'] = L.groupby('group').i.transform('size')
    R = d.loc[L.i]
    rep = pd.DataFrame({
        'group': L.group.values, 'group_size': L['size'].values, 'type': L.type.map(TYPES).values,
        'action': L.action.map(ACTION_UA).values, 'Record_ID': R.Record_ID.values,
        'Record_origin': R.Record_origin.values, 'ANZSRC_4D': R.ANZSRC_4D.values, 'Title': R.Title.values,
        'DOI': R.DOI.values, 'Abstract_before': R.Abstract.str[:400].values,
        'title_abstract_match': R._match.round(2).values})
    rep.to_pickle(os.path.join(outdir, 'step2_log.pkl'))
    json.dump({'rows_in': len(d), 'rows_out': len(out), 'deleted': len(drop), 'blanked': len(blank),
               'groups': len(groups)}, open(os.path.join(outdir, 'summary.json'), 'w'), indent=1)
    print(f'in {len(d)}  deleted {len(drop)}  abstracts removed {len(blank)}  out {len(out)}  groups {len(groups)}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
