#!/bin/bash
# Run all 3 benchmark phases in parallel with separate Neo4j containers.
# Usage: ./scripts/run_parallel.sh [--clean]
set -e

CLEAN_FLAG=""
if [ "$1" = "--clean" ]; then
    CLEAN_FLAG="--clean"
    echo "Clean mode: will wipe checkpoints and graph data"
fi

NEO4J_IMAGE="neo4j:5"
NEO4J_PASS="benchmark123"

declare -A PHASES
PHASES[controlled]=7687
PHASES[pipeline]=7688
PHASES[pipeline_presplit]=7689

echo "=== Starting Neo4j containers ==="
for phase in "${!PHASES[@]}"; do
    port=${PHASES[$phase]}
    name="neo4j-$phase"
    if docker ps -q -f name="$name" 2>/dev/null | grep -q .; then
        echo "  $name already running on port $port"
    else
        docker rm -f "$name" 2>/dev/null || true
        docker run -d --name "$name" -p "$port":7687 \
            -e NEO4J_AUTH="neo4j/$NEO4J_PASS" "$NEO4J_IMAGE" >/dev/null
        echo "  Started $name on port $port"
    fi
done

echo "=== Waiting for Neo4j readiness ==="
for phase in "${!PHASES[@]}"; do
    name="neo4j-$phase"
    for i in $(seq 1 30); do
        if docker exec "$name" cypher-shell -u neo4j -p "$NEO4J_PASS" "RETURN 1" 2>/dev/null | grep -q "1"; then
            echo "  $name ready"
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo "  ERROR: $name failed to start"
            exit 1
        fi
        sleep 2
    done
done

echo "=== Running benchmarks in parallel ==="
mkdir -p results
PIDS=()
for phase in "${!PHASES[@]}"; do
    port=${PHASES[$phase]}
    PYTHONUNBUFFERED=1 .venv/bin/python -m src.benchmark_runner "$phase" --port "$port" $CLEAN_FLAG \
        > "results/${phase}_output.log" 2>&1 &
    PIDS+=($!)
    echo "  Started $phase (PID $!, port $port)"
done

echo "=== Waiting for all phases to complete ==="
FAILED=0
for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
        echo "  PID $pid completed successfully"
    else
        echo "  PID $pid FAILED (exit code $?)"
        FAILED=1
    fi
done

echo ""
echo "=== Results ==="
ls -la results/benchmark_*.json 2>/dev/null || echo "  No results files found"
echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "All phases complete!"
else
    echo "Some phases failed. Check results/*_output.log for details."
    exit 1
fi
