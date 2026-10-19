#!/bin/sh
# Runs inside the heart-report container (see Dockerfile). Produces the DOCX, the PDF and
# page previews for visual checking.
set -eu
cd /work
node reports/build/build_report.js
cd reports
soffice --headless --convert-to pdf --outdir . Heart_Disease_MLOps_Report.docx >/dev/null 2>&1
echo "PDF: reports/Heart_Disease_MLOps_Report.pdf ($(pdfinfo Heart_Disease_MLOps_Report.pdf | awk '/^Pages/ {print $2}') pages)"
rm -rf build/preview && mkdir -p build/preview
pdftoppm -png -r 70 Heart_Disease_MLOps_Report.pdf build/preview/page
echo "Previews: reports/build/preview/"
