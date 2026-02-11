```markdown
# B4 Docker Deployment 🐳

## Quick Start

### Build the Docker image:

```bash
docker build -f projects/b4_macro_regimes/docker/Dockerfile -t quantforge/b4-regimes:latest .
```

### Run with Docker Compose:

```bash
docker-compose -f projects/b4_macro_regimes/docker/docker-compose.yml up
```

### Run directly:

```bash
docker run --rm \
  -e FRED_API_KEY=your_api_key_here \
  -v "%cd%\data\regimes:/app/data/regimes" \
  quantforge/b4-regimes:latest
```

## Requirements

- Docker Desktop for Windows (installed and running)
- FRED API key set in environment

## Output

Results are saved to `data/regimes/regime_labels_*.csv`
```