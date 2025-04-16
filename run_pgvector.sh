#!/bin/bash

# Check if container already exists
if [ "$(docker ps -a -q -f name=pgvector)" ]; then
  # Check if container is running
  if [ "$(docker ps -q -f name=pgvector)" ]; then
    echo "PgVector container is already running."
  else
    echo "Starting existing PgVector container..."
    docker start pgvector
  fi
else
  # Container doesn't exist, create it
  echo "Creating new PgVector container..."
  docker run -d \
    -e POSTGRES_DB=ai \
    -e POSTGRES_USER=ai \
    -e POSTGRES_PASSWORD=ai \
    -e PGDATA=/var/lib/postgresql/data/pgdata \
    -v pgvolume:/var/lib/postgresql/data \
    -p 5532:5432 \
    --name pgvector \
    agnohq/pgvector:16
fi

# Verify container is running
echo "PgVector container status:"
docker ps | grep pgvector
