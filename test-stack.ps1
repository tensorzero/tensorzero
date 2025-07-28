#!/usr/bin/env pwsh

# Simple test script to verify TensorZero is working correctly

Write-Host "Testing TensorZero Stack..." -ForegroundColor Green
Write-Host ""

$passed = 0
$total = 3

# Test Gateway Health
Write-Host "Testing Gateway health..." -ForegroundColor Blue
try {
    $response = Invoke-RestMethod -Uri "http://localhost:3000/health" -TimeoutSec 5
    if ($response.gateway -eq "ok" -and $response.clickhouse -eq "ok") {
        Write-Host "✅ Gateway health check passed" -ForegroundColor Green
        Write-Host "   Gateway: $($response.gateway)" -ForegroundColor White
        Write-Host "   ClickHouse: $($response.clickhouse)" -ForegroundColor White
        $passed++
    } else {
        Write-Host "❌ Gateway health check failed: Unexpected response" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Gateway health check failed: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test ClickHouse Access
Write-Host "Testing ClickHouse access..." -ForegroundColor Blue
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8123/ping" -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ ClickHouse is accessible" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "❌ ClickHouse access failed: Status $($response.StatusCode)" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ ClickHouse access failed: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test UI Access
Write-Host "Testing UI access..." -ForegroundColor Blue
try {
    $response = Invoke-WebRequest -Uri "http://localhost:4000" -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ UI is accessible" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "❌ UI access failed: Status $($response.StatusCode)" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ UI access failed: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Summary
Write-Host "Test Results: $passed/$total tests passed" -ForegroundColor Cyan

if ($passed -eq $total) {
    Write-Host "All tests passed! TensorZero is ready to use." -ForegroundColor Green
    Write-Host ""
    Write-Host "Access your services:" -ForegroundColor Cyan
    Write-Host "   • UI Dashboard:   http://localhost:4000" -ForegroundColor White
    Write-Host "   • Gateway API:    http://localhost:3000" -ForegroundColor White
    Write-Host "   • ClickHouse:     http://localhost:8123" -ForegroundColor White
    exit 0
} else {
    Write-Host "Some tests failed. Check the services and try again." -ForegroundColor Yellow
    exit 1
}
