#!/bin/bash

# Start Arq worker for document processing
# This script starts the background worker that processes documents

echo "Starting Arq worker for document processing..."
echo "Worker will process up to 4 documents concurrently"
echo ""

# Run the Arq worker
arq app.workers.arq_config.WorkerSettings
