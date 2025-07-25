#!/usr/bin/env pwsh

# TensorZero Launch Script
# This script starts the complete TensorZero stack (ClickHouse, Gateway, and UI)

Write-Host "Starting TensorZero Stack..." -ForegroundColor Green
Write-Host ""

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "ERROR: Docker is not running. Please start Docker Desktop and try again." -ForegroundColor Red
    exit 1
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "WARNING: .env file not found. Copying from .env.example..." -ForegroundColor Yellow
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "Created .env file from template. Please edit it with your API keys." -ForegroundColor Yellow
        Write-Host "Edit the .env file and run this script again." -ForegroundColor Yellow
        exit 1
    } else {
        Write-Host "ERROR: No .env.example found. Please create a .env file manually." -ForegroundColor Red
        exit 1
    }
}

# Check if config directory exists
if (-not (Test-Path "config")) {
    Write-Host "ERROR: Config directory not found. Please ensure config/tensorzero.toml exists." -ForegroundColor Red
    exit 1
}

# Pull latest images
Write-Host "Pulling latest Docker images..." -ForegroundColor Blue
docker-compose pull

# Start the services
Write-Host "Starting services..." -ForegroundColor Blue
docker-compose up -d

# Wait a moment for services to start
Start-Sleep -Seconds 5

# Check service status
Write-Host ""
Write-Host "Service Status:" -ForegroundColor Green
docker-compose ps

Write-Host ""
Write-Host "TensorZero Stack is starting up!" -ForegroundColor Green
Write-Host ""
Write-Host "Access Points:" -ForegroundColor Cyan
Write-Host "   * ClickHouse:     http://localhost:8123" -ForegroundColor White
Write-Host "   * Gateway API:    http://localhost:3000" -ForegroundColor White
Write-Host "   * UI Dashboard:   http://localhost:4000" -ForegroundColor White
Write-Host ""
Write-Host "Useful Commands:" -ForegroundColor Cyan
Write-Host "   * View logs:      docker-compose logs -f" -ForegroundColor White
Write-Host "   * Stop services:  docker-compose down" -ForegroundColor White
Write-Host "   * Restart:        docker-compose restart" -ForegroundColor White
Write-Host ""

# Wait for services to be healthy
Write-Host "Waiting for services to be healthy..." -ForegroundColor Yellow

$maxWait = 120  # 2 minutes
$waited = 0
$interval = 5

do {
    Start-Sleep -Seconds $interval
    $waited += $interval
    
    try {
        # Check services directly using docker ps
        $runningServices = docker-compose ps --format json | ConvertFrom-Json
        $clickhouseHealthy = ($runningServices | Where-Object { $_.Service -eq "clickhouse" -and $_.State -eq "running" -and $_.Health -eq "healthy" }).Count -gt 0
        $gatewayHealthy = ($runningServices | Where-Object { $_.Service -eq "gateway" -and $_.State -eq "running" -and $_.Health -eq "healthy" }).Count -gt 0
        
        if ($clickhouseHealthy -and $gatewayHealthy) {
            Write-Host "All services are healthy!" -ForegroundColor Green
            Write-Host "Opening UI in browser..." -ForegroundColor Blue
            Start-Process "http://localhost:4000"
            break
        }
    } catch {
        # If health check fails, just continue waiting
    }
    
    if ($waited -ge $maxWait) {
        Write-Host "Services are taking longer than expected to start." -ForegroundColor Yellow
        Write-Host "Check logs with: docker-compose logs" -ForegroundColor White
        break
    }
    
    Write-Host "Still waiting... ($waited/$maxWait seconds)" -ForegroundColor Gray
} while ($true)

Write-Host ""
Write-Host "TensorZero is ready! Happy building!" -ForegroundColor Green
