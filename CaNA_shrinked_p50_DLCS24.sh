#!/bin/bash

# ============================
# CaNA Shrinked Processing Pipeline
# Docker Container Activation
# ============================
echo "🚀 Starting CaNA (Cancer Analysis) Docker container for shrinking..."
cd "$(dirname "$0")"  # Go to script directory

# Remove existing container if it exists
docker rm -f cana_pipeline 2>/dev/null || true

# Start container using the PiNS medical imaging image
echo "📦 Launching ft42/pins:latest container..."
docker run -d --name cana_pipeline \
  -v "$(pwd):/app" \
  -w /app \
  ft42/pins:latest \
  tail -f /dev/null

# Create output directory and set proper permissions
echo "📁 Setting up output directories and permissions..."
docker exec cana_pipeline mkdir -p /app/demofolder/output/CaNA_shrinked_50_output
docker exec cana_pipeline chmod 777 /app/demofolder/output/CaNA_shrinked_50_output

# Install additional dependencies if needed
echo "🔧 Installing missing Python packages if needed..."
docker exec cana_pipeline pip install nibabel scikit-image > /dev/null 2>&1 || echo "⚠️  Some packages may already be installed"

echo "✅ Docker container is running with all dependencies"

# ============================
# Run CaNA Shrinking Pipeline
# ============================
echo "🔬 Running CaNA (Cancer Analysis) lung nodule shrinking processing..."

docker exec cana_pipeline python CaNA_LungNoduleSize_shrinked.py \
  --json_path ./demofolder/data/Experiments_DLCSD24_512xy_256z_771p25m_dataset.json \
  --dict_to_read "training" \
  --data_root ./demofolder/data/ \
  --lunglesion_lbl 23 \
  --scale_percent 50 \
  --log_file /app/demofolder/output/CaNA_shrinking_50.log \
  --save_dir /app/demofolder/output/CaNA_shrinked_50_output \
  --random_seed 42 \
  --prefix Aug23s50_ \
  --csv_output /app/demofolder/output/CaNA_shrinking_50_stats.csv

# ============================
# Cleanup and Results
# ============================
if [ $? -eq 0 ]; then
    echo "✅ CaNA shrinking processing completed successfully!"
    echo "📊 Check ./demofolder/output/ directory for results:"
    echo "   - Processing log: CaNA_shrinking_50.log"
    echo "   - Shrinked masks: CaNA_shrinked_50_output/"
    echo "   - Statistics CSV: CaNA_shrinking_50_stats.csv"
    echo "   - File prefix: Aug23s75_"
else
    echo "❌ CaNA shrinking processing failed. Check the logs above for errors."
fi

# Stop and remove container
echo "🧹 Cleaning up Docker container..."
docker stop cana_pipeline > /dev/null 2>&1
docker rm cana_pipeline > /dev/null 2>&1

echo "🎉 CaNA shrinking pipeline execution complete!"