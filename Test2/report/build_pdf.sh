#!/bin/bash
# Build Test2_Report.pdf from REPORT.md: pandoc -> self-contained HTML -> Chromium (Playwright) print.
# Run from Test2/. Needs pandoc and Node's playwright with Chromium (preinstalled in the build environment).
set -e
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
pandoc REPORT.md -f markdown+gfm_auto_identifiers+lists_without_preceding_blankline-yaml_metadata_block-implicit_figures \
  --mathml --standalone --embed-resources --css report/report.css --metadata pagetitle="ENG 6010 Test 2 Report" -o "$TMP/r.html"
# figure captions: paragraphs that consist only of an italic "Figure ..." line
python3 - "$TMP/r.html" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p, encoding="utf-8").read()
s = re.sub(r'<p><em>(Figure [^<]*(?:<[^/][^>]*>[^<]*</[^>]+>[^<]*)*)</em></p>', r'<p class="caption">\1</p>', s)
open(p, "w", encoding="utf-8").write(s)
PY
node report/to_pdf.js "$TMP/r.html" "$PWD/Test2_Report.pdf"
rm -rf "$TMP"
echo "-> Test2_Report.pdf"
