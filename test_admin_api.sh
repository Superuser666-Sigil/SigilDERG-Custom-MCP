#!/bin/bash
# Quick test script for Admin API endpoints

BASE_URL="http://127.0.0.1:8765"

echo "=== Sigil MCP Admin API Test Script ==="
echo ""

echo "1. Root endpoint (API info):"
curl -s "$BASE_URL/" | python3 -m json.tool
echo ""

echo "2. Server Status:"
curl -s "$BASE_URL/admin/status" | python3 -m json.tool
echo ""

echo "3. Index Statistics:"
curl -s "$BASE_URL/admin/index/stats" | python3 -m json.tool
echo ""

echo "4. Configuration (first 50 lines):"
curl -s "$BASE_URL/admin/config" | python3 -m json.tool | head -50
echo ""

echo "5. Recent Logs (last 10 lines):"
curl -s "$BASE_URL/admin/logs/tail?lines=10" | python3 -m json.tool
echo ""

echo "6. Rebuild Index (POST - all repos):"
echo "   (Uncomment to run)"
# curl -X POST "$BASE_URL/admin/index/rebuild" -H "Content-Type: application/json" | python3 -m json.tool
echo ""

echo "7. Rebuild Vector Index (POST):"
echo "   (Uncomment to run)"
# curl -X POST "$BASE_URL/admin/vector/rebuild" -H "Content-Type: application/json" | python3 -m json.tool
echo ""


