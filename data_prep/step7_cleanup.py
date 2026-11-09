#!/usr/bin/env python3
"""
ANZSRC dataset cleaning, step 7: records that do not describe research, near-identical texts,
and formatting left in titles and abstracts.

    python step7_cleanup.py <dataset_step6.csv> <title_only_step6.csv> <non_english_step6.csv> <output_dir>

Needs: pandas, ftfy (pip install ftfy).

1. Removed from the main file (CATEGORIES below):
   oral-history interview clips (title = a person's name, text = a transcript; one collection-level label),
   lockdown diary entries (Oxford), cartoons by one author, personal images, Nature "Daily briefing" news,
   journal webcast notices, "Editors' Best of" selections, thanks to reviewers, advertising, a letter
   posted as "1.pdf", and artworks whose notice forbids use for AI training / machine learning.
2. Moved: "This paper has no abstract. Cite this article as ..." -> title-only file (Abstract = '',
   Has_usable_abstract = '0'); one record whose text is Ukrainian once the English citation in front of
   it is ignored -> non-English file.
3. Near-identical texts: 127 groups found by TF-IDF (cosine >= 0.70) plus near-identical titles, kept when
   >= 80% of the words match in sequence or >= 90% of 3-word shingles of the shorter text are shared, and
   3 more groups that the same check found once the text had been cleaned (groups 128-130).
   Each group was reviewed by hand (GROUPS):
     K  versions, parts, files or translations of one item -> one record kept, the others removed
     O  different works that carry one template description -> the description stays with one record,
        the others keep only their title and move to the title-only file
     A  different works, each text has its own content -> all kept
     D  not research (collection pages, a publisher's template) -> removed
     M  mixed (actions per record)
   Within a group the record kept is the one whose title best matches the text, then the original
   dataset before repository records, then the longer text, then the one with a DOI.
4. Text fixed in place (Title, Abstract) in the main file, and titles in the title-only file:
   HTML entities and tags, broken encoding (ftfy, CP437 sequences, lost characters), "Abstract:" /
   "Summary" labels and "[Extract]" markers at the start, a citation or a fixed series preamble placed
   before the abstract,
   copyright / licence / publisher lines, links and e-mail addresses, "????" runs, file extensions in
   titles. Edits are local; every other character stays as it was.
All other columns are never changed; rows keep their order; moved rows are appended to the target file.
"""
import html, json, os, re, sys, unicodedata
import pandas as pd
import ftfy

FTFY = dict(unescape_html=False, uncurl_quotes=False, fix_latin_ligatures=False, fix_character_width=False,
            fix_line_breaks=False, normalization=None, remove_control_chars=False)

# ---------- markup and encoding ----------
ENTITY = re.compile(r'&(?:#\d{1,7}|#[xX][0-9a-fA-F]{1,6}|[a-zA-Z][a-zA-Z0-9]{1,31});')
TAG_NAMES = (r'i|b|u|em|strong|sub|sup|sc|smallcaps|small|big|tt|span|div|a|font|inf|scp|ital|bold|italic|italics|'
             r'p|br|hr|li|ul|ol|h[1-6]|blockquote|abstract|title|sec|label|list|list-item|'
             r'inline-formula|disp-formula|tex-math|alternatives|underline|jats:[a-z\-]+|mml:[a-z\-]+')
TAG = re.compile(rf'</?(?:{TAG_NAMES})\b[^<>]{{0,300}}?/?>', re.I)
BLOCK_TAG = re.compile(r'(?:\s*</?(?:p|br|hr|li|ul|ol|h[1-6]|div|blockquote|sec|jats:p|jats:sec|jats:title)\b[^<>]{0,300}?/?>\s*)+', re.I)
MOJI = re.compile(r'Ã[\x80-\xBF]|â€|Â[\xa0-\xbf]|Î[\xb1-\xbf]')
MOJI_LEFT = {'Â±': '±', 'Â°': '°', 'Âµ': 'µ', 'Â·': '·', 'Â§': '§', 'Â©': '©', 'Â®': '®', 'Â²': '²', 'Â³': '³',
             'Â½': '½', 'Â¼': '¼', 'Â¾': '¾'}
CP437_CONT = 'ÇüéâäàåçêëèïîìÄÅÉæÆôöòûùÿÖÜ¢£¥₧ƒáíóúñÑªº¿⌐¬½¼¡«»░▒▓│┤╡╢╖╕╣║╗╝╜╛┐'
CP437 = re.compile(f'[┬├─┼╞╟╚╔╩╦╠═╬╧╨╤╥╙╘╒╓╫╪┘┌█▄▌▐▀][{CP437_CONT}]|[αßΓπΣσµτΦΘΩδ∞φε∩][{CP437_CONT}]{{2}}')
LETTER = r'[^\W\d_]'
STRAY_MARK = re.compile('(?<![^\\W\\d_])[̀-ͯ]+')


def _cp437(m):
    try:
        return m.group().encode('cp437').decode('utf-8')
    except UnicodeError:
        return m.group()


def _fix_fffd(s):
    s = re.sub(r'�\s*(?=(?:19|20)\d{2}\b)', '© ', s)
    s = re.sub(rf'(?<={LETTER})�(?=(?:s|t|ll|re|ve|d|m)\b)', '’', s)                      # possessive / contraction
    s = re.sub(rf'(?<!{LETTER})�(?={LETTER})|(?<={LETTER}|[.,;:!?])�(?!{LETTER})', "'", s)  # quote at a word edge
    return s.replace('�', ' ')


def fix_markup(s):
    """entities, HTML tags, mojibake, lost characters"""
    rules = []
    t = ENTITY.sub(lambda m: re.sub(r'[\r\x0b\x0c]', ' ', html.unescape(m.group())), s)
    if t != s:
        rules.append('html_entity')
    u = BLOCK_TAG.sub(lambda m: '\n' if '\n' in m.group() else ' ', t)
    u = TAG.sub('', u)
    if u != t:
        rules.append('html_tag')
    v = u
    if MOJI.search(v):
        v = ftfy.fix_text(v, **FTFY)
        for a, b in MOJI_LEFT.items():
            v = v.replace(a, b)
    if re.search(r'(?<=\S)Â(?= )', v):  # NBSP mojibake whose NBSP became a plain space ("etÂ al.")
        v = re.sub(r'(?<=\S)Â(?= )', '', v)
    if CP437.search(v):
        v = CP437.sub(_cp437, v)
        if re.search('[̀-ͯ]', v):
            v = STRAY_MARK.sub('', unicodedata.normalize('NFC', v))
    if '�' in v:
        v = _fix_fffd(v)
    if v != u:
        rules.append('mojibake')
    return v, rules


# ---------- local deletion helper ----------
WS = ' \t '


def delete_spans(s, spans):
    """remove [a, b) spans; fix only the gap each removal leaves (double spaces, ' .', '()', '. .')"""
    if not spans:
        return s
    pieces, pos = [], 0
    for a, b in sorted(spans):
        a = max(a, pos)
        pieces.append(s[pos:a]); pos = max(pos, b)
    pieces.append(s[pos:])
    res = pieces[0]
    for right in pieces[1:]:
        left = res
        L, R = left.rstrip(WS), right.lstrip(WS)
        lgap, rgap = left[len(L):], right[:len(right) - len(R)]
        if L.endswith(('(', '[')) and R.startswith((')', ']')):
            L, R = L[:-1].rstrip(WS), R[1:]
            lgap, rgap = ' ', ''
        if R[:1] in ('.', ',', ';', ':') and L[-1:] in ('.', ',', ';', ':', '!', '?'):
            R = R[1:].lstrip(WS) if R[1:2] in tuple(WS) else R[1:]
        if not L or not R or R[:1] in ('.', ',', ';', ':', '!', '?', ')', ']', '\n') or L[-1:] == '\n':
            res = L + R
        else:
            res = L + (lgap or rgap or ' ')[:1] + R
    return res


# ---------- leading labels / markers ----------
LEAD_MARK = re.compile(r'^\W*\[(?:extract|no abstract, showing first paragraph)\]\s*', re.I)
LEAD_QUOTED = re.compile(r'^\W*abstract\s*:\s*"', re.I)
LEAD_LABEL = re.compile(r'^[\s\W]*?(?i:abstract|summary(?:\s+statement)?)'
                        r'(?:\s*[:.\-–—]+\s*|\s+(?=[A-Z0-9“"‘\'(])|(?=[A-Z][a-z]|[A-Z]{2}|\d))')
LEAD_GLUED = re.compile('^[\\s\\W]*?(?:ABSTRACT|Abstract)\u200b?(?=[‘“"\'$\u0391-\u03a9\u03b1-\u03c9\u200b])\u200b?')
LEAD_SPECIAL = re.compile(r'^\W*(?:abstract\s*/\s*tuhinga whakarāpopoto|abstract\s*&\s*project overview)\s*', re.I)
PREFIX_BOILER = re.compile(
    r'^\W*Electronic journal\s*"Modern problems of science and education",\s*Issue No\.?\s*\d+,\s*\d{4}\.?\s*'
    r'(?:\((?:[^()]|\([^()]*\))*\)\.?\s*)?'
    r"|^\W*This metadata record and it[’']?s attached files make statements about the kinds of data collected as part of "
    r'this research, and set out policies for governance of that data, now and in the future\.\s*Description\s*:\s*')
SUMMARY_KEEP = re.compile(r'^\W*summary\s+(?:of|on|for|and|statistics|table|report|points?)\b', re.I)


def strip_lead(s):
    rules = []
    t = LEAD_MARK.sub('', s, count=1)
    if t != s:
        rules.append('extract_marker')
    if LEAD_QUOTED.match(t):
        t2 = LEAD_QUOTED.sub('', t, count=1).rstrip()
        if t2.endswith('"') and t2.count('"') % 2 == 1:
            t2 = t2[:-1].rstrip()
        rules.append('lead_label'); t = t2
    elif LEAD_SPECIAL.match(t) or LEAD_GLUED.match(t):
        t2 = (LEAD_SPECIAL if LEAD_SPECIAL.match(t) else LEAD_GLUED).sub('', t, count=1)
        if len(t2) > 20:
            rules.append('lead_label'); t = t2
    elif not SUMMARY_KEEP.match(t):
        t2 = LEAD_LABEL.sub('', t, count=1)
        if t2 != t and len(t2) > 20:
            rules.append('lead_label'); t = t2
    return t, rules


def strip_prefix_boiler(s):
    m = PREFIX_BOILER.match(s)
    if not m or len(s) - m.end() < 100:
        return s, []
    return s[m.end():], ['boilerplate_prefix']


CITE_PREFIX = re.compile(r'^\W*(?:to cite this|how to cite|cite this|citation\s*:)', re.I)
CITE_MARK = re.compile(r'(?:-{3,}\s*)?\b(?:Abstract|Description)\s*(?:[:\-–—]+|(?=[A-Z]))\s*')


def strip_citation(s):
    if not CITE_PREFIX.match(s):
        return s, []
    m = CITE_MARK.search(s, 20)
    if not m or m.start() > 600:
        return s, []
    return s[m.end():], ['citation_prefix']


# ---------- copyright / licence / publisher lines ----------
PUBLISHER = (r'Elsevier|Wiley|Springer|Taylor\s*(?:&|and)\s*Francis|Informa|Emerald|SAGE|MDPI|Frontiers|BMJ|Cell Press|'
             r'Oxford University Press|Cambridge University Press|Routledge|Brill|De Gruyter|Karger|Thieme|Lippincott|Wolters|'
             r'Nature|IOP|AIP|APS|ACS|IEEE|Royal Society|Hindawi|PLOS|Dove|Bentham|Inderscience|IGI|Copernicus|PeerJ|JMIR|'
             r'Mary Ann Liebert|Annual Reviews|CSIRO|Mohr|Edward Elgar|Palgrave|Macmillan|John Wiley|Blackwell|Pergamon|Academic Press')
RIGHTS = re.compile(
    r'all rights reserved|creative commons|open[- ]access article|distributed under the terms|this is an open[- ]access|'
    r'protected by copyright|psycinfo database record|for permissions,? please|reproduced with (?:the )?permission|'
    r'copyright and licen[cs]e information|originally published in\b[^.]{0,80}\(https?://|\blicensee\b|rights are reserved|'
    rf'^\s*(?:this (?:article|paper|work|chapter) (?:is|was) )?published by\b|\bpublished by (?:{PUBLISHER})',
    re.I)
COPY_START = re.compile(r'^\s*[“"‘\']?\s*(?:copyright\s*(?::\s*)?(?:©|\(c\)|(?=(?:19|20)\d{2}))|©|\(c\)\s*(?=(?:19|20)\d{2}))', re.I)
COPY_STRONG = re.compile(
    rf'(?:{PUBLISHER})|\b(?:author|authors|author\(s\)|et al|press|society|association|ltd|limited|inc|llc|gmbh|b\.v|'
    r'periodicals|publishing|publications|publisher|rights|reserved|licen[cs]ed|majesty|crown)\b', re.I)
COPY_WEAK = re.compile(r'\b(?:university|college|academy|journal|government|institute|council|foundation|trust|federation|'
                       r'media|ministry|department|organi[sz]ation|commonwealth|state of)\b', re.I)
BARE_COPY = re.compile(r'^\s*(?:©|\(c\))\s*(?:19|20)\d{2}\s*[,.]?\s*', re.I)
ABBR = re.compile(r'\b(?:e\.g|i\.e|Dr|Prof|No|Vol|pp|vs|St|Mt|U\.S|U\.K|Jr|Sr)\.')
BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z©(\[“"‘\'])|(?<![Cc]opyright)(?<!COPYRIGHT)(?<![Cc]opyright:)\s+(?=©|\([cC]\)\s*(?:19|20)\d{2}|Copyright\s*(?:©|\([cC]\)|\d{4}))')
RIGHTS_HINT = re.compile(r'©|\(c\)|copyright|rights reserved|creative commons|open[- ]access|published by|publishing|psycinfo|licen[cs]', re.I)


def _sentences(s):
    """[(start, end)] of sentence-like pieces; abbreviations such as 'B.V.' do not end a sentence"""
    prot = ABBR.sub(lambda m: m.group()[:-1] + '\x00', s)
    spans, pos = [], 0
    for m in BOUNDARY.finditer(prot):
        spans.append((pos, m.start())); pos = m.end()
    spans.append((pos, len(s)))
    return [(a, b) for a, b in spans if b > a]


RIGHTS_LEAD = re.compile(
    r'^\W*(?:all rights reserved|this (?:is an|article is an?) open[- ]access|'
    r'this (?:article|work|paper|dataset|record|item|chapter|book) is (?:licensed|distributed|published|protected|copyrighted)|'
    r'psycinfo database record|creative commons|licensed under|originally published in|for permissions|'
    r'reproduced with (?:the )?permission|published by\b|the complete bibliographic information|'
    r'personal use of this material is permitted|permission from ieee must be obtained|'
    r"this is an? (?:author[’']?s? )?(?:accepted manuscript|original manuscript|pre-?print|post-?print)\b)", re.I)
MID_PUB = re.compile(rf'^[^.]{{0,80}}\bpublished by\b[^.]{{0,60}}(?:{PUBLISHER})', re.I)
JOURNAL_PUB = re.compile(r'\bpublished by\b[^.]{0,80}\bon behalf of\b', re.I)


def _is_rights(sent):
    w = len(sent.split())
    if w > 60:
        return False
    if RIGHTS_LEAD.match(sent) or (w <= 40 and JOURNAL_PUB.search(sent)):
        return True
    if COPY_START.match(sent):
        return bool(w <= 6 or (w <= 25 and COPY_STRONG.search(sent)) or (w <= 12 and COPY_WEAK.search(sent)))
    return bool(w <= 20 and RIGHTS.search(sent))


def strip_rights(s):
    if not RIGHTS_HINT.search(s):
        return s, []
    sp = _sentences(s)
    txt = [s[a:b] for a, b in sp]
    n = len(sp)
    rights = [_is_rights(t) for t in txt]
    holder = [bool(COPY_START.match(t) and len(t.split()) <= 40) for t in txt]
    drop = [False] * n
    i = 0
    while i < n and (rights[i] or (holder[i] and i + 1 < n and rights[i + 1])):
        drop[i] = True; i += 1
    j = n - 1
    while j >= i and (rights[j] or (holder[j] and j + 1 < n and drop[j + 1])):
        drop[j] = True; j -= 1
    for k in range(i, j + 1):  # a copyright line inside the text (e.g. before "Basic Protocol 1: ...")
        if rights[k] and ((COPY_START.match(txt[k]) and len(txt[k].split()) <= 30)
                          or (MID_PUB.match(txt[k]) and len(txt[k].split()) <= 12)):
            drop[k] = True
    spans = [sp[k] for k in range(n) if drop[k]]
    out = delete_spans(s, spans) if spans else s
    m = BARE_COPY.match(out)
    if m:
        out = out[m.end():]
    if out == s or len(out.split()) < 15:
        return s, []
    return out, ['copyright_licence']


# ---------- links, e-mails, lost characters ----------
URL = re.compile(r'(?:https?://|\bwww\.)[^\s<>"\]\)]*[^\s<>"\]\).,;:!?\'’”]')
EMAIL = re.compile(r'\b[\w.+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b')
QRUN = re.compile(r'\?{4,}')


def strip_links(s):
    rules, spans = [], []
    for name, rx in (('url', URL), ('email', EMAIL), ('qmark_run', QRUN)):
        found = [m.span() for m in rx.finditer(s)]
        found = [f for f in found if not any(a <= f[0] < b for a, b in spans)]
        if found:
            rules.append(name); spans += found
    if not spans:
        return s, []
    return delete_spans(s, spans), rules


# ---------- whole fields ----------
FILE_EXT = re.compile(r'(?:\s*\.(?:pdf|docx?|xlsx?|xlsm|csv|tsv|zip|rar|7z|gz|tar|pptx?|txt|rtf|odt|ods|png|jpe?g|tiff?|gif|bmp|svg|'
                      r'mp4|mov|avi|mkv|mp3|wav|m4a|sav|dta|rds|rdata|json|xml|nc|h5|hdf5|kmz|kml|shp|ipynb|epub|html?))+\s*$', re.I)


def clean_abstract(s):
    if not s:
        return s, []
    rules = []
    t, r = fix_markup(s); rules += r
    t, r = strip_citation(t); rules += r
    t, r = strip_prefix_boiler(t); rules += r
    t, r = strip_lead(t); rules += r
    t, r = strip_rights(t); rules += r
    t, r = strip_lead(t); rules += [x for x in r if x not in rules]
    t, r = strip_links(t); rules += r
    t = t.strip()
    if t == s:
        return s, []
    return t, rules


def clean_title(s):
    if not s:
        return s, []
    t, rules = fix_markup(s)
    u = FILE_EXT.sub('', t)
    if u != t and len(u.strip()) >= 3:
        if '_' in u and ' ' not in u.strip():
            u = u.replace('_', ' ')
        rules.append('filename_title'); t = u
    t = t.strip()
    if t == s:
        return s, []
    return t, rules


# ======================= records that are not research =======================
def categories(d):
    T, A = d.Title, d.Abstract
    rules = [
        ('oral_history_clip', A.str.match(r'This (?:clip|quote) is from a series of 100 life history interviews')),
        ('lockdown_diary', d.DOI.str.startswith('10.25446/oxford') & T.str.match(r'^\d{5}: ')),
        ('cartoon', (d.Authors == 'Partridge, Matthew') & d.Publication_type.str.contains('Figure')),
        ('personal_upload', T.isin(['Bible annotation', 'Wanted Poster drawing', 'A note from a friend', 'Sketch of an eye'])
         & d.Publication_type.str.contains('Figure')),
        ('news_digest', T.str.startswith('Daily briefing:')),
        ('webcast', T.str.contains(r'\bWebcast\b') & A.str.match(r'This \d+-minute webcast')),
        ('editorial_matter', T.str.match(r"Editors[’'] Best of \d{4}")),
        ('reviewer_thanks', T.str.contains(r'(?i)\backnowledg\w*\b.{0,80}\breviewers\b|\bthank(?:s| you)\b.{0,40}\breviewers\b')),
        ('spam_promo', T.str.startswith('WholesalePrinted Boxes') | T.str.startswith('Unlocking the Power of Hashnode')
         | T.str.startswith('Passey Electric') | A.str.contains(r'(?i)welcome to our youtube channel')),
        ('letter', (T == '1.pdf') & A.str.startswith('Dear Sir/Madam')),
        ('no_ai_training', A.str.contains('CRUCIAL INTELLECTUAL PROPERTY NOTICE') & A.str.contains('AI training')),
        ('no_abstract', A.str.match(r'This paper has no abstract')),
    ]
    cat = pd.Series('', index=d.index)
    for name, m in rules:
        cat[m & cat.eq('')] = name
    return cat


TO_NON_ENGLISH = {'mat_7751a03073faf9e62841'}  # Ukrainian chapter; only the citation in front of it is English

# ======================= near-identical groups (reviewed by hand) =======================
# (group, type, {Record_ID: action}); action K = keep, D = delete, B = keep the title only (-> title-only file)
GROUPS = [
    (1, 'K', {'mat_d5747c6ab2a4b7c1504d': 'D', 'mat_0c0ef27d6141cd8cf9a0': 'D', 'mat_fd8f232bf2f9a9acfc6a': 'K', 'mat_41d9fc1a188c8e641965': 'D', 'mat_8a16c241e6207e24b8a1': 'D', 'mat_d43f0d35fb21a1ce9bde': 'D', 'mat_9a118a21c268d138e707': 'D', 'mat_ca2976c72c73b21108e8': 'D', 'mat_46533f4243014daf7bab': 'D', 'mat_c5f48d368337bdc00226': 'D', 'mat_3a73b201d08bf97fb112': 'D', 'mat_0b4412eac573d8d4ccbf': 'D', 'mat_898f75fafe470c8218b6': 'D', 'mat_e4da768e6c99423d5340': 'D', 'mat_c49d8d4310a2e5921695': 'D', 'mat_5ed8377c9f63bb273ab2': 'D', 'mat_358e896eb49ffe53248b': 'D', 'mat_fd0d00329065c4fadab2': 'D', 'mat_a934e7fc72fc218777d2': 'D', 'mat_d2a4ee22c2a23117a127': 'D', 'mat_1bb53062fdac69b0e699': 'D', 'mat_a07046fdc78920f2c73a': 'D'}),
    (2, 'K', {'mat_160bcee04bc05321833b': 'K', 'mat_5095df64f60633bf2880': 'D', 'mat_f35409572c2a34ea8e0d': 'D', 'mat_b60e58aefdd331e121a7': 'D', 'mat_58aa3e3d5cd90d87edf4': 'D', 'mat_f175f17afcd70cd61611': 'D', 'mat_b2b7d77911959409169d': 'D', 'mat_0db586888ea177a5a8b8': 'D'}),
    (3, 'O', {'mat_b591a43ff6229623f90a': 'B', 'mat_0152a22b3684f97f078a': 'K', 'mat_f4c7d55483bf7c4d4540': 'B', 'mat_ba25de3436a3a59679ab': 'B', 'mat_8c32da3d97fabdbd773c': 'B', 'mat_b3ada6a1477a1eaecf7a': 'B', 'mat_49fe43644aa20cab362a': 'B', 'mat_9c6ee19e482955b900f4': 'B'}),
    (4, 'K', {'mat_88b670013dcb2d329a88': 'K', 'mat_7e5ae7de344d5e281684': 'D', 'mat_41b74caaee0ad83ed4ee': 'D', 'mat_2c946f87e9e5f621395f': 'D', 'mat_71f09a88e970364726d6': 'D', 'mat_7d3c92519f3bab01344c': 'D', 'mat_79c6bce81fde41ff7bea': 'D', 'mat_7b4517cffc277b3b6ec2': 'D'}),
    (5, 'K', {'mat_88dc63f9ad88ea822d22': 'D', 'mat_dece7fd3e81e01abd791': 'K', 'mat_8b37f2f28ea063445af5': 'D', 'mat_0ff67fbe0571be2c8c93': 'D', 'mat_1e3dd52b6fad95689558': 'D', 'mat_e357e3a813c10a0b4984': 'D', 'mat_03b5bf8a3af1e9dbb0fc': 'D'}),
    (6, 'K', {'mat_bdb5c111a1b0c55bdc80': 'D', 'mat_7fc6386fcd09376b34e0': 'K', 'mat_16c0bed991e699998c87': 'D', 'mat_86a2e579c23abad65ff1': 'D', 'mat_64c05d0f5a8baa5a2ab7': 'D', 'mat_da9afe438f6651926bd8': 'D'}),
    (7, 'M', {'art_a29fa0d543800b064b73': 'D', 'mat_4d32417de121ef1db995': 'D', 'mat_c86a3adf7462006539f0': 'D', 'art_5bfac9203171e4098f4d': 'K', 'art_6865de35afd6822934e4': 'K'}),
    (8, 'A', {'art_9b8db9e8532106a1e9d5': 'K', 'art_2df78d2939c4d4231aaf': 'K', 'art_53e98551696252d40067': 'K', 'art_d77cbac6867ae6ef7042': 'K', 'art_7bc2a20db9a8c6dc3a77': 'K'}),
    (9, 'K', {'mat_39dd9be95d3f9b7781ed': 'K', 'mat_5e3a9157cddfe250ceb3': 'D', 'mat_d42bb7a9ce36eb5b8217': 'D', 'mat_4c38e45aa9bb462c365c': 'D'}),
    (10, 'K', {'mat_104cc608b43fb18bf509': 'K', 'mat_b272fc7c7f913b3f38fc': 'D', 'mat_87c9b12fd85a866ebea6': 'D', 'mat_7284809eb5ae85a1f03e': 'D'}),
    (11, 'K', {'mat_1d489dbadc3110fb6db6': 'K', 'mat_de1967e15d570f50d126': 'D', 'mat_fa713d0bc355f18b84d5': 'D', 'mat_7a50e561a80eeb32b6ff': 'D'}),
    (12, 'K', {'mat_4375dc7c5c6a71137d4a': 'K', 'mat_bd5909caa52808aabbae': 'D', 'mat_69b9fcb877aedb881f1a': 'D', 'mat_258fe17c0286e6aca787': 'D'}),
    (13, 'K', {'mat_ebcce530f380d9991732': 'K', 'mat_086eaf8cbd0ef32535b4': 'D', 'mat_6bfa233c85837b3a30e1': 'D', 'mat_65ccce158bbde5e5dd8a': 'D'}),
    (14, 'D', {'mat_a1276b7711391ef151e4': 'D', 'mat_1da0a9d6b74df451b5fb': 'D', 'mat_952a46ab3d691ce5d5b0': 'D', 'mat_67f43efc4b5c19c812f8': 'D'}),
    (15, 'K', {'mat_d4cb5f40aac0ac9a0ce4': 'K', 'mat_816dc2a6da280237a8b4': 'D', 'mat_565d144d1feedb6b0a73': 'D'}),
    (16, 'K', {'mat_90c9891c1e613a695dbd': 'D', 'mat_99ee863c3dbab1c748ec': 'K', 'mat_4a7d6d0369ed6cf3af31': 'D'}),
    (17, 'K', {'mat_0cfb47c587a47aafedc8': 'D', 'mat_87c8bc7052a91cb95228': 'K', 'mat_071f438cc0504bfb8f41': 'D'}),
    (18, 'O', {'mat_191a33f6f678b147aa70': 'K', 'mat_050f1043cd228687efb0': 'B', 'mat_ddc3a9ed588e66f4ff97': 'B'}),
    (19, 'K', {'mat_d74ba209d37d0396793d': 'K', 'mat_e46a6b5ceed1ea8f0c62': 'D', 'mat_000f04a7e86279ec5cb9': 'D'}),
    (20, 'A', {'mat_4a138ba5648dea86e800': 'K', 'mat_daed48fd93c3c282ef68': 'K', 'mat_bd49ef020f2ddc22c2b3': 'K'}),
    (21, 'O', {'mat_f3256cd93e3dffef3353': 'K', 'mat_20f3c95d59e6b17b18b7': 'B', 'mat_80d68e42a5d20f469176': 'B'}),
    (22, 'M', {'mat_c0d73421f31f103ae9de': 'K', 'mat_26dc53987d735ac9acc3': 'D', 'mat_0be5c1aa526c2e717ff9': 'B'}),
    (23, 'K', {'mat_fa9112f297d83cac24f8': 'D', 'mat_588637f143bb8fb66624': 'K', 'mat_da0444184540b66bb07a': 'D'}),
    (24, 'K', {'orig_0000589': 'K', 'orig_0000592': 'D'}),
    (25, 'K', {'orig_0000794': 'K', 'art_b52341f537086898017d': 'D'}),
    (26, 'A', {'orig_0006594': 'K', 'orig_0007329': 'K'}),
    (27, 'K', {'orig_0015398': 'K', 'orig_0015581': 'D'}),
    (28, 'K', {'orig_0017602': 'K', 'mat_98323169ee22db6cf723': 'D'}),
    (29, 'K', {'orig_0024535': 'K', 'orig_0024566': 'D'}),
    (30, 'K', {'orig_0043713': 'K', 'mat_4a6b2e20139c453e187b': 'D'}),
    (31, 'A', {'orig_0081193': 'K', 'orig_0081263': 'K'}),
    (32, 'K', {'orig_0083530': 'D', 'orig_0083535': 'K'}),
    (33, 'K', {'orig_0084964': 'D', 'orig_0084989': 'K'}),
    (34, 'K', {'art_6ff8a417b57849148ad4': 'D', 'art_f69082dd44663508e965': 'K'}),
    (35, 'K', {'art_bd57240ca157e735d46c': 'K', 'art_09715c38d85cbee9a767': 'D'}),
    (36, 'K', {'art_3fdef45ce247b1bdcac2': 'D', 'art_93f51ec687f2ef6b55f3': 'K'}),
    (37, 'K', {'art_4bf7ce676edeb721da47': 'D', 'art_987cbba176cebe3e150c': 'K'}),
    (38, 'K', {'art_c48d27122cd70f9692c0': 'K', 'mat_b694fae39050c1bb4b66': 'D'}),
    (39, 'K', {'art_842817fb28540fecf6e0': 'K', 'mat_dc06c6ea67104c44752e': 'D'}),
    (40, 'K', {'art_346e859cc3b482904e77': 'D', 'art_d227e4cbdc02f5fe5dca': 'K'}),
    (41, 'K', {'art_745c7696546693c8dcb9': 'K', 'mat_fa7d6a49e1bcedc8f650': 'D'}),
    (42, 'K', {'art_5f0cac44c1f3d87b9d6b': 'D', 'mat_057867702b4941130876': 'K'}),
    (43, 'O', {'art_d36dfdce57b43595be04': 'K', 'art_a703338e94b57353b82c': 'B'}),
    (44, 'K', {'mat_3c5ea407ae8a71bdce02': 'D', 'mat_49815e10b7fa18123d09': 'K'}),
    (45, 'K', {'mat_ed54197c0083d19a25bc': 'K', 'mat_0806167af62f10e2a66f': 'D'}),
    (46, 'K', {'mat_d924190bdba4d562f946': 'K', 'mat_e1677278203cb9d4a0bd': 'D'}),
    (47, 'K', {'mat_1648398ddd00d826de59': 'K', 'mat_c4ef0db3bb8d3e6f0c5d': 'D'}),
    (48, 'K', {'mat_cf7f9e1fc0ea16b45955': 'D', 'mat_efc209643df303277563': 'K'}),
    (49, 'K', {'mat_6993e5c5ee274befdf1e': 'K', 'mat_b9e1c65a6cb6c931d9c3': 'D'}),
    (50, 'K', {'mat_7028cea655eebad9ad07': 'K', 'mat_252b84991d5e2f221161': 'D'}),
    (51, 'K', {'mat_27f08493f64d9aa96eb7': 'K', 'mat_3f283d5324089fdb173a': 'D'}),
    (52, 'K', {'mat_0884e4529dd076b0d309': 'K', 'mat_bd1c6cf5479958f591be': 'D'}),
    (53, 'K', {'mat_215e9cddadbc8445999a': 'K', 'mat_15bb2a54488a28c43595': 'D'}),
    (54, 'K', {'mat_299662cbda32216ac0b5': 'D', 'mat_e709c6cf4a967076bb2f': 'K'}),
    (55, 'K', {'mat_fe56c557897d5e085ffa': 'K', 'mat_3a72a3886516672d293e': 'D'}),
    (56, 'K', {'mat_9e565494a91416aad53c': 'K', 'mat_a5a7b36631081db4cc4d': 'D'}),
    (57, 'K', {'mat_0637aca9c41317fdc2ec': 'D', 'mat_a1858d1d22cf8470705c': 'K'}),
    (58, 'A', {'mat_a36797a85914b2aa142a': 'K', 'mat_9f5009f4b3df3c58dcc1': 'K'}),
    (59, 'K', {'mat_b274072bdc71ad091174': 'K', 'mat_d5d0947b8882b1e655a6': 'D'}),
    (60, 'K', {'mat_0c153a668adb8502d40f': 'K', 'mat_6a5d029fa3fbed79adb8': 'D', 'mat_6df9b16a61f99e63ea41': 'D'}),
    (61, 'O', {'mat_128cb9a46d6d3031f7dd': 'K', 'mat_57419404e0273c56095b': 'B'}),
    (62, 'K', {'mat_0a34fe0c7879303bba94': 'K', 'mat_d24015a0141f476dc86a': 'D'}),
    (63, 'K', {'mat_08b3b77f35c85522dd2b': 'K', 'mat_9718257f41a206d843d0': 'D'}),
    (64, 'K', {'mat_444d195cb754504f57f9': 'K', 'mat_2422a5658f876cad9479': 'D'}),
    (65, 'K', {'mat_7b62b7016b56b822d130': 'K', 'mat_91b382607440b75cf808': 'D'}),
    (66, 'K', {'mat_7e00b7bff52445ac9dc8': 'D', 'mat_800725f2003ba3074a4f': 'K'}),
    (67, 'O', {'mat_390c83ea94be0f23d707': 'K', 'mat_9215154177cd4e1eb734': 'B'}),
    (68, 'K', {'mat_04366488cf97f43aa937': 'D', 'mat_d7607fd94dc2f0249a1f': 'K'}),
    (69, 'K', {'mat_6913d312eed2e777be95': 'K', 'mat_aa78a4b4de463413092f': 'D'}),
    (70, 'K', {'mat_acf3e55c0fec22a07617': 'D', 'mat_f031a649331e4654d1ee': 'K'}),
    (71, 'K', {'mat_eae285a11361b4517578': 'K', 'mat_f1c66d8de577d8e6b631': 'D'}),
    (72, 'K', {'mat_c2cba3fe09be752c8571': 'K', 'mat_70010f7f2e46d6172427': 'D'}),
    (73, 'O', {'mat_e2f58141715723a99c42': 'K', 'mat_853dd3956fc71c2f5732': 'B'}),
    (74, 'K', {'mat_a722b69868bbcfcb44b9': 'K', 'mat_54c84949ff19d6e4b518': 'D'}),
    (75, 'K', {'mat_969c629961a850f6a495': 'K', 'mat_d36ff3ec5b4ef08c00c2': 'D'}),
    (76, 'K', {'mat_87f801b5182300ae4e2e': 'K', 'mat_99680c8b8c910a6d9d97': 'D'}),
    (77, 'K', {'mat_3750e33cfb24c13d1367': 'K', 'mat_a3783161e26b7c590777': 'D'}),
    (78, 'K', {'mat_c01904c9d07df4f3ea61': 'K', 'mat_1e98f1121e993ae32206': 'D'}),
    (79, 'K', {'mat_327d2a8f140a30d60042': 'K', 'mat_396a6727eb2e21cd5dca': 'D'}),
    (80, 'K', {'mat_195f2fba82f41e79ee5e': 'D', 'mat_e0ba616bd48f60ef4cf0': 'K'}),
    (81, 'K', {'mat_e050be420326e847718c': 'K', 'mat_b51a6e5127ff88b4c35b': 'D'}),
    (82, 'K', {'mat_63ea0ecd0487bea3077a': 'D', 'mat_3ebb03cb5ed4cdbf0eed': 'K'}),
    (83, 'O', {'mat_99bd69959a0fc99df4d8': 'B', 'mat_14c49f13ed264dacef4a': 'K'}),
    (84, 'K', {'mat_5235d5df60824e89ce2c': 'K', 'mat_df7e4d61c17c4b594574': 'D'}),
    (85, 'K', {'mat_4e66c9596a0bee5bf9ac': 'K', 'mat_857e6af41b44a1110d6f': 'D'}),
    (86, 'K', {'mat_61547e16748e956490cc': 'K', 'mat_6959b1c541eeffce6ae5': 'D'}),
    (87, 'K', {'mat_fb2615d0ae3e2ffe266d': 'D', 'mat_f32b6e34fc9f080aad16': 'K'}),
    (88, 'K', {'mat_dccef9665b252937acf7': 'D', 'mat_3582002a62044acb0308': 'K'}),
    (89, 'K', {'mat_3119303beb82834e7dab': 'K', 'mat_370d4c02c719ee02f2f9': 'D'}),
    (90, 'K', {'mat_0e093c2074ca6816e095': 'K', 'mat_54d3ed134c3c6965f824': 'D'}),
    (91, 'O', {'mat_5ba059bf987b41129116': 'K', 'mat_6abfadef54fb6b4df48e': 'B'}),
    (92, 'K', {'mat_bef8b92d26df01129163': 'K', 'mat_29b942068fd17e19e110': 'D'}),
    (93, 'K', {'mat_f058ff3d706547de1d29': 'K', 'mat_ea7b9f4ec1aa06f31447': 'D'}),
    (94, 'K', {'mat_7da8932fad4f6ca6b3b9': 'K', 'mat_a2c1dea5230f5f9b38ad': 'D'}),
    (95, 'K', {'mat_8c064caf8fdfa9f10887': 'K', 'mat_1a45a1b715b115dfafbc': 'D'}),
    (96, 'K', {'mat_4a1ceaff665f5ba106ff': 'K', 'mat_98e551b1b9bdbbb7f77d': 'D'}),
    (97, 'K', {'mat_9a8c747602e6f05abb5c': 'K', 'mat_0e0e5e9412b57c1dc819': 'D'}),
    (98, 'K', {'mat_b7106541298eedbda0e4': 'D', 'mat_574f2b3da7ccff7b431e': 'K'}),
    (99, 'K', {'mat_ea706b0ba01cd9bbfed1': 'K', 'mat_0685d5b8e4b353f5d830': 'D'}),
    (100, 'O', {'mat_12108ba2c6e55006a6c1': 'B', 'mat_63cea37d06fbbe02ba4f': 'K'}),
    (101, 'O', {'mat_58a5e2ef622e22aa695b': 'K', 'mat_b8eecc35c16bb7e915b4': 'B'}),
    (102, 'K', {'mat_db7789b51816906110a1': 'K', 'mat_b2fe9def0e2b8929426b': 'D'}),
    (103, 'K', {'mat_1c26565df9e2da2aaa0f': 'K', 'mat_3e3002993d026fad4f34': 'D'}),
    (104, 'K', {'mat_431b57db74d9cf63e278': 'D', 'mat_b6e85461e0e1b5b88a8b': 'K'}),
    (105, 'K', {'mat_3fbe90a6f69991b7aaf6': 'K', 'mat_7f6cedde8d880e7fc974': 'D'}),
    (106, 'K', {'mat_1beb5ce57ff3557e12e4': 'K', 'mat_76884a9cbfb0ad7ab270': 'D'}),
    (107, 'D', {'mat_cebce0f2065aa4ee4601': 'D', 'mat_5fb4e776f589dffe0b62': 'D'}),
    (108, 'K', {'mat_d52c6f4ccd086b2b9bb6': 'D', 'mat_c445dd89dba310a032fa': 'K'}),
    (109, 'K', {'mat_b07fa2c2487fe089a6c3': 'D', 'mat_3ada3653182e59d67674': 'K'}),
    (110, 'K', {'mat_b9544c01c1a983dd9626': 'K', 'mat_bd88d0ec60f8e3e5f924': 'D'}),
    (111, 'O', {'mat_8dd263a208030610565d': 'K', 'mat_f92be057f2fc829bfea7': 'B'}),
    (112, 'O', {'mat_6b888d83c6761f600516': 'K', 'mat_17594c50d6f0c7b625b9': 'B'}),
    (113, 'K', {'mat_7949fc48d70336210016': 'K', 'mat_ac6654975fad7a426efc': 'D'}),
    (114, 'O', {'mat_0cc7d2943828a2fbdaa4': 'K', 'mat_039c6e237c9f291e5941': 'B'}),
    (115, 'O', {'mat_30ee45e8618336de5dea': 'B', 'mat_57fb6ca9c061a19fdffb': 'K'}),
    (116, 'K', {'mat_a655a89457755e927238': 'K', 'mat_5581a820f2db7a1e8cae': 'D'}),
    (117, 'K', {'mat_30ebb63739d0a42f95fc': 'K', 'mat_645f8ebdc97180669cad': 'D'}),
    (118, 'K', {'mat_1d3b9606c2d25d987f57': 'K', 'mat_fa88064d5fead2957791': 'D'}),
    (119, 'K', {'mat_06be0485eb3349bdab62': 'K', 'mat_31f4138cbd20e9803c99': 'D'}),
    (120, 'K', {'mat_7821758160b73615e76e': 'D', 'mat_4b6390ad390f98ab4187': 'K'}),
    (121, 'K', {'mat_2333a5cc194b253d150a': 'K', 'mat_57b909db8cfac588269e': 'D'}),
    (122, 'K', {'mat_4ea6346ff2874709b4bd': 'K', 'mat_03270bf6507c32ea3a10': 'D'}),
    (123, 'K', {'mat_133323d3352df9f30d0a': 'K', 'mat_7d6e94fc52c50cf62c72': 'D'}),
    (124, 'K', {'mat_72d5d972c32625f649f1': 'D', 'mat_4417ab7c8f2f10d1a243': 'K'}),
    (125, 'K', {'mat_872fd2f7963abb0b3114': 'K', 'mat_e15994825cc5d067dfd3': 'D'}),
    (126, 'O', {'mat_379785d0402211f6afdf': 'K', 'mat_d22f17988a659022f051': 'B'}),
    (127, 'K', {'mat_3b1d0f7ac23a5d6bec30': 'D', 'mat_e48002c429d4ccd57e22': 'K'}),
    (128, 'K', {'mat_e43c2190b559b0d92084': 'K', 'mat_988777e6cada96355a46': 'D', 'mat_d066f2a038f287035e10': 'D'}),
    (129, 'K', {'mat_9918e70182f9dafee7ee': 'K', 'mat_99038e86bf1023f26470': 'D'}),
    (130, 'K', {'mat_479eab5a36707746ff19': 'K', 'mat_c4219f2492d5c07cca23': 'D'}),
]
GROUP_TYPE = {'K': 'versions_parts', 'O': 'shared_template', 'A': 'distinct_works', 'D': 'not_research', 'M': 'mixed'}


def main(inp, title_inp, nonen_inp, outdir):
    os.makedirs(outdir, exist_ok=True)
    rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    d, t, ne = rd(inp), rd(title_inp), rd(nonen_inp)
    cols = list(d.columns)
    assert list(t.columns) == cols and list(ne.columns) == cols

    cat = categories(d)
    rid = dict(zip(d.Record_ID, d.index))
    action, gmeta = {}, {}
    for g, typ, acts in GROUPS:
        for r, a in acts.items():
            assert r in rid, (g, r)
            action[rid[r]] = a; gmeta[rid[r]] = (g, typ, len(acts))
    overlap = [i for i in action if cat[i] != '']
    assert not overlap, overlap

    delete = cat.ne('') & cat.ne('no_abstract')
    delete |= pd.Series([action.get(i) == 'D' for i in d.index], index=d.index)
    to_title = cat.eq('no_abstract') | pd.Series([action.get(i) == 'B' for i in d.index], index=d.index)
    to_nonen = d.Record_ID.isin(TO_NON_ENGLISH)
    keep = ~delete & ~to_title & ~to_nonen

    # text fixes (kept main records; titles of the title-only file)
    log_txt = []
    main_out = d[keep].copy()
    for i in main_out.index:
        for f, fn in (('Title', clean_title), ('Abstract', clean_abstract)):
            new, rules = fn(main_out.at[i, f])
            if rules:
                log_txt.append((main_out.at[i, 'Record_ID'], 'main', f, ';'.join(rules), main_out.at[i, f], new))
                main_out.at[i, f] = new
    t_out = t.copy()
    for i in t_out.index:
        new, rules = clean_title(t_out.at[i, 'Title'])
        if rules:
            log_txt.append((t_out.at[i, 'Record_ID'], 'title_only', 'Title', ';'.join(rules), t_out.at[i, 'Title'], new))
            t_out.at[i, 'Title'] = new
    moved_t = d[to_title].copy()
    moved_t['Abstract'] = ''
    moved_t['Has_usable_abstract'] = '0'
    for i in moved_t.index:  # moved titles get the same title fixes
        new, rules = clean_title(moved_t.at[i, 'Title'])
        if rules:
            log_txt.append((moved_t.at[i, 'Record_ID'], 'title_only', 'Title', ';'.join(rules), moved_t.at[i, 'Title'], new))
            moved_t.at[i, 'Title'] = new

    main_out[cols].to_csv(os.path.join(outdir, 'dataset_step7.csv'), index=False, encoding='utf-8-sig')
    pd.concat([t_out, moved_t[cols]], ignore_index=True).to_csv(os.path.join(outdir, 'title_only.csv'), index=False,
                                                                 encoding='utf-8-sig')
    pd.concat([ne, d[to_nonen][cols]], ignore_index=True).to_csv(os.path.join(outdir, 'non_english.csv'), index=False,
                                                                  encoding='utf-8-sig')

    # logs
    rows = []
    for i in d.index:
        if cat[i]:
            rows.append((i, 'category', cat[i], '', 'moved_title_only' if cat[i] == 'no_abstract' else 'deleted'))
        elif i in action:
            g, typ, n = gmeta[i]
            a = {'K': 'kept', 'D': 'deleted', 'B': 'moved_title_only'}[action[i]]
            rows.append((i, 'group', GROUP_TYPE[typ], g, a))
        elif to_nonen[i]:
            rows.append((i, 'language', 'ukrainian_after_citation', '', 'moved_non_english'))
    L = pd.DataFrame(rows, columns=['i', 'kind', 'reason', 'group', 'action'])
    R = d.loc[L.i]
    L = L.assign(Record_ID=R.Record_ID.values, Record_origin=R.Record_origin.values, ANZSRC_4D=R.ANZSRC_4D.values,
                 Publication_type=R.Publication_type.values, Title=R.Title.values, DOI=R.DOI.values,
                 Abstract_before=R.Abstract.str[:400].values)
    L.drop(columns='i').to_pickle(os.path.join(outdir, 'step7_records_log.pkl'))
    pd.DataFrame(log_txt, columns=['Record_ID', 'file', 'field', 'rules', 'before', 'after']).to_pickle(
        os.path.join(outdir, 'step7_text_log.pkl'))
    tl = pd.DataFrame(log_txt, columns=['Record_ID', 'file', 'field', 'rules', 'before', 'after'])
    rule_counts = {}
    for f, rs in zip(tl.file + ':' + tl.field, tl.rules):
        for r in rs.split(';'):
            rule_counts[f'{f}:{r}'] = rule_counts.get(f'{f}:{r}', 0) + 1
    s = {'main_in': len(d), 'title_in': len(t), 'non_english_in': len(ne),
         'deleted': int(delete.sum()), 'moved_title_only': int(to_title.sum()), 'moved_non_english': int(to_nonen.sum()),
         'main_out': int(keep.sum()), 'title_out': len(t) + int(to_title.sum()), 'non_english_out': len(ne) + int(to_nonen.sum()),
         'deleted_by_category': {k: int(v) for k, v in cat[delete & cat.ne('')].value_counts().items()},
         'group_records': len(action), 'groups': len(GROUPS),
         'group_actions': {k: int(v) for k, v in pd.Series(action).value_counts().items()},
         'text_fixed_main_records': int(tl[tl.file == 'main'].Record_ID.nunique()),
         'text_fixed_title_only_records': int(tl[tl.file == 'title_only'].Record_ID.nunique()),
         'text_rules': rule_counts}
    json.dump(s, open(os.path.join(outdir, 'summary.json'), 'w'), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
