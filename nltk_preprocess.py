import argparse
import os
import re
from collections import defaultdict

import nltk
import numpy as np
from docx import Document
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pypdf import PdfReader

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

LEMMA = WordNetLemmatizer()
STOP = set(stopwords.words("english"))


# ---------------------------------------------------------- Q1: read + preprocess
def read_document(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    if ext == ".docx":
        return "\n".join(p.text for p in Document(path).paragraphs)
    with open(path, encoding="utf-8") as f:
        return f.read()


def preprocess(text, verbose=False):
    """lowercase -> tokenize -> drop punctuation/numbers -> drop stopwords -> lemmatize"""
    stages = [("tokens", word_tokenize(text.lower()))]
    stages.append(("no punct/numbers", [t for t in stages[-1][1] if t.isalpha()]))
    stages.append(("no stopwords", [t for t in stages[-1][1] if t not in STOP]))
    stages.append(("normalized", [LEMMA.lemmatize(t) for t in stages[-1][1]]))
    if verbose:
        for name, toks in stages:
            print(f"[{name}] ({len(toks)} items) {toks[:25]}")
    return stages[-1][1]


def build_corpus(folder):
    docs = {}
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith((".txt", ".pdf", ".docx")):
            toks = preprocess(read_document(os.path.join(folder, fn)))
            if toks:
                docs[fn] = toks
            else:
                print(f"[skipped] {fn}: no extractable text")
    if not docs:
        raise SystemExit(f"No readable .pdf/.docx/.txt files in '{folder}'.")
    return docs


def norm(term):
    out = preprocess(term)
    return out[0] if out else term.lower()


# ------------------------------------------------ Boolean parser (NOT > AND > OR)
def evaluate(query, get, AND, OR, NOT):
    toks = re.findall(r"\(|\)|\w+", query)
    pos = 0

    def peek():
        return toks[pos].upper() if pos < len(toks) else None

    def take():
        nonlocal pos
        pos += 1
        return toks[pos - 1]

    def parse_or():
        left = parse_and()
        while peek() == "OR":
            take()
            left = OR(left, parse_and())
        return left

    def parse_and():
        left = parse_not()
        while peek() == "AND":
            take()
            left = AND(left, parse_not())
        return left

    def parse_not():
        if peek() == "NOT":
            take()
            return NOT(parse_not())
        if peek() == "(":
            take()
            v = parse_or()
            take()  # ')'
            return v
        return get(take())

    return parse_or()


# ---------------------------------------------- (a) Term-document incidence matrix
class IncidenceMatrix:
    def __init__(self, docs):
        self.doc_ids = list(docs)
        self.terms = sorted({t for toks in docs.values() for t in toks})
        self.row = {t: i for i, t in enumerate(self.terms)}
        self.M = np.zeros((len(self.terms), len(self.doc_ids)), dtype=int)
        for j, d in enumerate(self.doc_ids):
            for t in docs[d]:
                self.M[self.row[t], j] = 1

    def show(self):
        w = max(map(len, self.terms)) + 2
        print("term".ljust(w) + " ".join(d.rjust(5) for d in self.doc_ids))
        for t in self.terms:
            print(t.ljust(w) + " ".join(str(v).rjust(5) for v in self.M[self.row[t]]))

    def vec(self, term):
        t = norm(term)
        return self.M[self.row[t]] if t in self.row else np.zeros(len(self.doc_ids), dtype=int)

    def query(self, q):
        res = evaluate(q, self.vec, lambda a, b: a & b, lambda a, b: a | b, lambda a: 1 - a)
        return [d for d, v in zip(self.doc_ids, res) if v]


# ------------------------------------------------------------ (b) Inverted index
class InvertedIndex:
    def __init__(self, docs):
        self.doc_ids = list(docs)
        idx = defaultdict(set)
        for d, toks in docs.items():
            for t in toks:
                idx[t].add(d)
        self.postings = {t: sorted(s) for t, s in sorted(idx.items())}

    def show(self):
        for t, p in self.postings.items():
            print(f"{t:<18} df={len(p)}  ->  {p}")

    @staticmethod
    def intersect(p1, p2):  # merge algorithm for AND
        i = j = 0
        out = []
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                out.append(p1[i]); i += 1; j += 1
            elif p1[i] < p2[j]:
                i += 1
            else:
                j += 1
        return out

    def query(self, q):
        return evaluate(
            q,
            lambda t: self.postings.get(norm(t), []),
            self.intersect,
            lambda a, b: sorted(set(a) | set(b)),
            lambda p: [d for d in self.doc_ids if d not in p],
        )


# ------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", help="folder with .pdf/.docx/.txt files")
    ap.add_argument("--file", help="file for the Q1 demo (default: first document)")
    args = ap.parse_args()

    if not args.docs:
        raise SystemExit('Please include --docs, e.g.: python ir_boolean.py --docs "my doc folder"')
    if not os.path.isdir(args.docs):
        raise SystemExit(f"Folder '{args.docs}' not found. Check the path given to --docs.")

    docs = build_corpus(args.docs)

    print("=" * 70, "\nQ1  TEXT PRE-PROCESSING\n", "=" * 70, sep="")
    q1 = args.file or os.path.join(args.docs, next(iter(docs)))
    raw = read_document(q1)
    print(f"--- {q1} ---\nRAW: {raw[:300]!r}")
    preprocess(raw, verbose=True)

    print("\n" + "=" * 70 + f"\nQ2  CORPUS: {len(docs)} documents\n" + "=" * 70)
    for d, t in docs.items():
        print(d, t[:20], "..." if len(t) > 20 else "")

    print("\n" + "=" * 70 + "\n(a) TERM-DOCUMENT INCIDENCE MATRIX\n" + "=" * 70)
    im = IncidenceMatrix(docs)
    im.show()

    print("\n" + "=" * 70 + "\n(b) INVERTED INDEX\n" + "=" * 70)
    ii = InvertedIndex(docs)
    ii.show()

    print("\n" + "=" * 70 + "\nBOOLEAN QUERY PROCESSING\n" + "=" * 70)
    print("Type a Boolean query (AND / OR / NOT, parentheses). Blank line to quit.")
    try:
        while q := input("query> ").strip():
            print(f"  Incidence matrix : {im.query(q)}\n  Inverted index   : {ii.query(q)}")
    except EOFError:
        pass