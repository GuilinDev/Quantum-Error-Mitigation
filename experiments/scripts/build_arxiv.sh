#!/usr/bin/env bash
# Build an arXiv-ready source tarball from paper/.
# arXiv compiles with pdflatex but does not run bibtex, so main.bbl is included.
set -eu
cd "$(dirname "$0")/../../paper"

# fresh compile so main.bbl matches main.tex
pdflatex -interaction=nonstopmode main.tex > /dev/null
bibtex main > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null

STAGE=$(mktemp -d)
mkdir -p "$STAGE/figures" "$STAGE/tables"
cp main.tex main.bbl "$STAGE/"
cp tables/*.tex "$STAGE/tables/"
# only the figures the manuscript references
for f in overview_diagram budget_pareto phase_boundary scaling; do
  cp "figures/$f.pdf" "$STAGE/figures/"
done

# verify the staged source compiles standalone
( cd "$STAGE" && pdflatex -interaction=nonstopmode main.tex > compile.log 2>&1 \
  && pdflatex -interaction=nonstopmode main.tex > compile.log 2>&1 )
ERRS=$(grep -c "^!" "$STAGE/compile.log" || true)
PAGES=$(pdfinfo "$STAGE/main.pdf" | awk '/Pages/{print $2}')
rm -f "$STAGE"/main.pdf "$STAGE"/*.aux "$STAGE"/*.log "$STAGE"/*.out "$STAGE"/compile.log

OUT="$PWD/arxiv_submission.tar.gz"
tar -czf "$OUT" -C "$STAGE" .
rm -rf "$STAGE"
echo "arxiv tarball: $OUT (standalone compile: $ERRS errors, $PAGES pages)"
