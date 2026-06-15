#!/bin/bash
# Enterprise RAG CS Docker Entrypoint

echo "=== Enterprise RAG CS Starting ==="
echo "Python: $(python --version)"

# Create data directories
mkdir -p /app/logs /app/chroma_data /app/data

# Verify key modules
python -c "
import sys
modules = ['query_rewriter','hybrid_retriever','context_compressor',
           'hallucination_guard','safety_enforcer','escalation_handler',
           'evaluation','observability']
for m in modules:
    try:
        __import__(m)
        print(f'  {m}: OK')
    except Exception as e:
        print(f'  {m}: FAIL - {e}')
        sys.exit(1)
print('All modules loaded successfully')
"

# Start server
exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
