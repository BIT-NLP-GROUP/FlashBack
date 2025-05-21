python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $OUTPUT_DIR \
  --index $INDEX_DIR \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storeRaw