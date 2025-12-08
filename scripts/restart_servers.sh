# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

#!/usr/bin/env bash
# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com
#
# Script to stop and restart all Sigil MCP Server processes
# - MCP Server (Python)
# - Frontend Dev Server (Vite/Node)

set -euo pipefail

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Function to find and kill processes
kill_processes() {
    local pattern="$1"
    local name="$2"
    
    log_info "Checking for running $name processes..."
    
    # Find PIDs matching the pattern
    local pids
    pids=$(ps aux | grep -E "$pattern" | grep -v grep | awk '{print $2}' || true)
    
    if [ -z "$pids" ]; then
        log_info "No $name processes found"
        return 0
    fi
    
    log_info "Found $name processes: $pids"
    
    # Kill processes
    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            log_info "Stopping process $pid..."
            kill "$pid" 2>/dev/null || true
        fi
    done
    
    # Wait a moment for graceful shutdown
    sleep 2
    
    # Force kill if still running
    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            log_warning "Force killing process $pid..."
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    
    log_success "Stopped all $name processes"
}

# Function to check if port is in use
check_port() {
    local port="$1"
    if lsof -i ":$port" >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for port to be free
wait_for_port_free() {
    local port="$1"
    local timeout=10
    local elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        if ! check_port "$port"; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    
    return 1
}

# Stop all servers
log_info "=========================================="
log_info "Stopping all Sigil MCP Server processes"
log_info "=========================================="

# Kill MCP Server (Python)
kill_processes "python.*sigil_mcp\.server|python.*-m sigil_mcp\.server" "MCP Server"

# Kill Frontend Dev Server (Vite/Node)
kill_processes "vite.*sigil-admin-ui|node.*vite.*sigil-admin-ui" "Frontend Dev Server"

# Wait for ports to be free
log_info "Waiting for ports to be free..."
if check_port 8000; then
    log_warning "Port 8000 still in use, waiting..."
    wait_for_port_free 8000 || log_warning "Port 8000 may still be in use"
fi

if check_port 5173; then
    log_warning "Port 5173 still in use, waiting..."
    wait_for_port_free 5173 || log_warning "Port 5173 may still be in use"
fi

log_success "All processes stopped"
echo ""

# Start all servers
log_info "=========================================="
log_info "Starting all Sigil MCP Server processes"
log_info "=========================================="

# Check for virtual environment
if [ ! -d ".venv" ]; then
    log_error "Virtual environment not found at .venv"
    log_error "Please create it first: python3 -m venv .venv"
    exit 1
fi

# Activate virtual environment and use venv's Python directly
# This ensures the venv is used even with nohup
VENV_PYTHON=".venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    log_error "Python not found in virtual environment at $VENV_PYTHON"
    log_error "Please recreate the venv: python3 -m venv .venv"
    exit 1
fi

# Start MCP Server
log_info "Starting MCP Server on port 8000..."
if [ -f "/tmp/sigil_server.log" ]; then
    mv /tmp/sigil_server.log /tmp/sigil_server.log.old 2>/dev/null || true
fi

nohup "$VENV_PYTHON" -m sigil_mcp.server > /tmp/sigil_server.log 2>&1 &
MCP_PID=$!

# Wait a moment for server to start
sleep 3

# Check if MCP server started successfully
if ! kill -0 "$MCP_PID" 2>/dev/null; then
    log_error "MCP Server failed to start"
    log_error "Check logs: tail -50 /tmp/sigil_server.log"
    exit 1
fi

if check_port 8000; then
    log_success "MCP Server started (PID: $MCP_PID)"
else
    log_warning "MCP Server process started but port 8000 not listening yet"
fi

# Start Frontend Dev Server
log_info "Starting Frontend Dev Server on port 5173..."

if [ ! -d "sigil-admin-ui" ]; then
    log_warning "Frontend directory not found, skipping frontend server"
else
    cd sigil-admin-ui
    
    # Check for node_modules
    if [ ! -d "node_modules" ]; then
        log_warning "node_modules not found, installing dependencies..."
        npm install
    fi
    
    # Tell Vite which backend to use (defaults to MCP port env or 8000)
    export VITE_API_BASE_URL="http://127.0.0.1:${SIGIL_MCP_PORT:-8000}"
    
    if [ -f "/tmp/frontend.log" ]; then
        mv /tmp/frontend.log /tmp/frontend.log.old 2>/dev/null || true
    fi
    
    nohup npm run dev > /tmp/frontend.log 2>&1 &
    FRONTEND_PID=$!
    
    cd "$PROJECT_ROOT"
    
    # Wait a moment for frontend to start
    sleep 3
    
    # Check if frontend started successfully
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        log_warning "Frontend Dev Server may have failed to start"
        log_warning "Check logs: tail -50 /tmp/frontend.log"
    elif check_port 5173; then
        log_success "Frontend Dev Server started (PID: $FRONTEND_PID)"
    else
        log_warning "Frontend process started but port 5173 not listening yet"
    fi
fi

echo ""
log_success "=========================================="
log_success "All servers started successfully"
log_success "=========================================="
echo ""
log_info "MCP Server:     http://127.0.0.1:8000"
log_info "Admin UI:       http://localhost:5173"
echo ""
log_info "Logs:"
log_info "  MCP Server:   tail -f /tmp/sigil_server.log"
log_info "  Frontend:     tail -f /tmp/frontend.log"
echo ""
log_info "To stop all servers, run:"
log_info "  $0 --stop"
echo ""

# Handle --stop flag
if [ "${1:-}" = "--stop" ]; then
    log_info "Stopping all servers..."
    kill_processes "python.*sigil_mcp\.server|python.*-m sigil_mcp\.server" "MCP Server"
    kill_processes "vite.*sigil-admin-ui|node.*vite.*sigil-admin-ui" "Frontend Dev Server"
    log_success "All servers stopped"
    exit 0
fi
