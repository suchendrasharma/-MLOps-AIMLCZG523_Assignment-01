# Report toolchain: Node (docx generator) + LibreOffice (DOCX -> PDF) + Poppler (page previews).
#   docker build -t heart-report reports/build
#   docker run --rm -v "${PWD}:/work" heart-report            # writes reports/*.docx and *.pdf
FROM node:22-bookworm-slim

# fonts-crosextra-carlito is metric-compatible with Calibri, so the PDF keeps Word's layout
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libreoffice-writer-nogui poppler-utils fonts-crosextra-carlito fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /tools
COPY package.json ./
RUN npm install --omit=dev --no-audit --no-fund

WORKDIR /work
ENV NODE_PATH=/tools/node_modules
CMD ["sh", "/work/reports/build/build.sh"]
