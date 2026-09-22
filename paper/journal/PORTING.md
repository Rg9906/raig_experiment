# Porting the manuscript to another venue

`main.tex` is written against `IEEEtran` in journal mode. That is a
venue-neutral choice made for a practical reason: TinyTeX has the class
installed locally, so the document compiles on the author's machine and every
table can be checked against `results/*.json` by `audit_paper.py`. Nothing in
the content assumes an IEEE venue.

The body sections (`sec_*.tex`) are class-agnostic apart from the items listed
below. Porting is an edit to `main.tex` plus the handful of fixes here.

## What is deliberately not used

These packages are **not** installed in the local TinyTeX tree and are avoided
throughout, so a port does not have to unpick them:

- `multirow` — tables use `\cmidrule` groupings and repeated labels instead.
- `subcaption` / `subfig` — there are no subfigures.
- `threeparttable` — table notes live in the caption.
- `siunitx` — units are written out ("thirty-second", "ms/turn").
- `amsthm` — a three-line `proof` environment is declared in `main.tex`
  because `IEEEtran` does not provide one and `amsthm` would fight the class's
  theorem styling. Most other classes *do* want `amsthm`; see below.

## Elsevier (`elsarticle`)

1. `\documentclass[preprint,review,12pt]{elsarticle}`.
2. Replace the `\author`/`\thanks` block with `\author[a]{...}` and
   `\affiliation[a]{organization={...}, ...}`.
3. Replace `\markboth{...}{...}` — no equivalent; delete it.
4. Abstract: `elsarticle` wants `\begin{frontmatter} ... \end{frontmatter}`
   wrapping `\title`, `\author`, `\begin{abstract}`, and `\begin{keyword}`.
   `sec_abstract.tex` currently emits `abstract` + `IEEEkeywords`; change
   `IEEEkeywords` to `keyword` and move the `\input` inside `frontmatter`.
5. Delete `\IEEEPARstart{A}{recurring}` at the top of `sec_intro.tex` and
   restore the plain word ("A recurring").
6. Load `amsthm` and delete the local `proof` environment from `main.tex`.
7. Bibliography: `\bibliographystyle{elsarticle-num}`.
8. `\begin{table*}` becomes `\begin{table}` in a one-column preprint layout;
   `tab:main` is the only starred float.

## ACM (`acmart`, e.g. TOIS / TIST)

1. `\documentclass[acmsmall]{acmart}` (TOIS/TIST use `acmsmall`).
2. Author block: `\author{}`, `\affiliation{}`, `\email{}`; add
   `\renewcommand{\shortauthors}{Saran Vishakan}`.
3. Abstract goes before `\maketitle`; keywords use `\keywords{...}`, and
   `acmart` also wants CCS concepts (`\begin{CCSXML}...`). Nearest concepts:
   *Information systems → Users and interactive retrieval*, and
   *Computing methodologies → Active learning settings*.
4. Delete `\IEEEPARstart` and `\markboth` as above.
5. `acmart` loads `amsthm` itself: delete the local `proof` environment, and
   replace `\newtheorem{proposition}{Proposition}` etc. with `acmart`'s
   pre-defined `proposition`, `corollary`, `definition` styles (declaring them
   again is an error).
6. `booktabs` is already loaded by `acmart`; the duplicate `\usepackage` is
   harmless but can go.
7. Bibliography: `\bibliographystyle{ACM-Reference-Format}`.
8. `acmart` in `acmsmall` is single-column: change `table*` to `table`.

## Springer (`sn-jnl`)

1. `\documentclass[pdflatex,sn-mathphys]{sn-jnl}`.
2. Front matter uses `\author*[1]{\fnm{}\sur{}}`, `\affil[1]{\orgdiv{}...}`.
3. Same `\IEEEPARstart` / `\markboth` deletions; same `amsthm` note.
4. Bibliography: `\bibliographystyle{sn-basic}`.

## Double-blind submission

Replace the whole `\author`/`\thanks` block with an anonymous one and remove
the repository URL `\thanks`. The URL is the only identifying string outside
the author block; `grep -n "repository URL" main.tex` finds it.

## After any port

Run, from `paper/journal/`:

    pdflatex main && bibtex main && pdflatex main && pdflatex main

then, from the repository root (the audit resolves `results/` and
`paper/journal/` relative to the working directory, so it must be run there):

    py audit_paper.py

The audit does not depend on the document class — it reads the `.tex` sources
as text — so it should report the same pass count before and after the port. A
changed pass count means a table lost a number in the conversion.

One porting hazard specific to this manuscript: do **not** edit the `.tex`
files through a shell heredoc. The environment's heredoc collapses `\r` in
`\ref` into a carriage return, silently turning `Table~\ref{tab:x}` into
`Table~<CR>ef{tab:x}`, which compiles without error and renders as `Table~ef`.
Use an editor or a Python file. `grep -n "ef{" *.tex` finds any survivors.
