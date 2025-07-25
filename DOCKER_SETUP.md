# TensorZero Docker Setup

This directory contains everything you need to run the complete TensorZero stack locally using Docker.

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed and running
- At least one API key configured (see Configuration section)

### Launch TensorZero

**Option 1: PowerShell (Recommended)**
```powershell
.\start-tensorzero.ps1
```

**Option 2: Batch file**
```cmd
start-tensorzero.bat
```

**Option 3: Manual Docker Compose**
```bash
docker-compose up -d
```

### Stop TensorZero

**PowerShell:**
```powershell
.\stop-tensorzero.ps1
```

**Manual:**
```bash
docker-compose down
```

## 🔧 Configuration

### Environment Variables (.env)
Copy `.env.example` to `.env` and configure your API keys:

```bash
# Required for the example configuration
OPENAI_API_KEY=sk-your-openai-key-here

# Optional - add other providers as needed
ANTHROPIC_API_KEY=your-anthropic-key-here
MISTRAL_API_KEY=your-mistral-key-here
# ... etc
```

### TensorZero Configuration (config/tensorzero.toml)
The `config/tensorzero.toml` file defines your functions, models, and variants. The included example provides a simple haiku generation function.

You can modify this file to add your own functions and model configurations.

## 📍 Access Points

Once running, you can access:

- **TensorZero UI**: http://localhost:4000
- **Gateway API**: http://localhost:3000
- **ClickHouse**: http://localhost:8123

## 🏗️ Architecture

The Docker setup includes:

### Services
1. **ClickHouse** (`tensorzero-clickhouse`)
   - Database for storing inference data
   - Port: 8123
   - Credentials: `chuser` / `chpassword`
   - Persistent data volume

2. **Gateway** (`tensorzero-gateway`)
   - Main API service
   - Port: 3000
   - Health endpoint: `/health`

3. **UI** (`tensorzero-ui`)
   - Web dashboard
   - Port: 4000
   - Connects to both Gateway and ClickHouse

### Networking
All services run on a custom Docker network (`tensorzero`) for isolated communication.

## 🔍 Useful Commands

### Viewing Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f gateway
docker-compose logs -f ui
docker-compose logs -f clickhouse
```

### Service Management
```bash
# Check status
docker-compose ps

# Restart specific service
docker-compose restart gateway

# Rebuild and restart
docker-compose up -d --build
```

### Data Management
```bash
# Stop and remove data volumes (⚠️ This will delete all data!)
docker-compose down -v

# Full cleanup
docker-compose down -v --rmi all --remove-orphans
```

## 🛠️ Troubleshooting

### Services Won't Start
1. Check Docker Desktop is running
2. Verify ports 3000, 4000, and 8123 are not in use
3. Check logs: `docker-compose logs`

### Gateway Health Check Fails
- Wait longer - the gateway needs ClickHouse to be ready first
- Check ClickHouse logs: `docker-compose logs clickhouse`
- Verify environment variables in `.env`

### UI Can't Connect
- Ensure Gateway is healthy: `docker-compose ps`
- Check Gateway logs: `docker-compose logs gateway`
- Verify TENSORZERO_GATEWAY_URL environment variable

### Permission Issues (Linux/Mac)
- Make scripts executable: `chmod +x *.ps1`
- Run with proper permissions: `sudo docker-compose up -d`

## 📚 Next Steps

1. **Configure Functions**: Edit `config/tensorzero.toml` to add your AI functions
2. **Add Providers**: Configure additional model providers in `.env`
3. **Explore the UI**: Visit http://localhost:4000 to explore the dashboard
4. **Test the API**: Make requests to http://localhost:3000

## 🔗 Resources

- [TensorZero Documentation](https://www.tensorzero.com/docs)
- [Configuration Guide](https://www.tensorzero.com/docs/gateway/configuration)
- [API Reference](https://www.tensorzero.com/docs/gateway/api)
- [Docker Compose Reference](https://docs.docker.com/compose/)
