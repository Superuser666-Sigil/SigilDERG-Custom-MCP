#!/usr/bin/env python3
# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
API Key Management Script for Sigil MCP Server.

Usage:
    python manage_auth.py generate    # Generate a new API key
    python manage_auth.py show        # Show current API key status
    python manage_auth.py reset       # Reset/regenerate API key
"""

import sys
from .auth import initialize_api_key, API_KEY_FILE


def generate():
    """Generate a new API key."""
    if API_KEY_FILE.exists():
        print(f"❌ API key already exists at {API_KEY_FILE}")
        print("   Use 'reset' to generate a new key")
        return 1
    
    api_key = initialize_api_key()
    if api_key:
        print("✅ API Key Generated Successfully!")
        print("=" * 60)
        print(f"API Key: {api_key}")
        print("=" * 60)
        print()
        print("⚠️  SAVE THIS KEY SECURELY - You won't see it again!")
        print()
        print("Set it in your environment:")
        print(f"  export SIGIL_MCP_API_KEY={api_key}")
        print()
        print("Or use it as a header in requests:")
        print(f"  X-API-Key: {api_key}")
        return 0
    
    return 1


def show():
    """Show current API key status."""
    if not API_KEY_FILE.exists():
        print("❌ No API key configured")
        print("   Run 'python manage_auth.py generate' to create one")
        return 1
    
    print("✅ API key is configured")
    print(f"   Location: {API_KEY_FILE}")
    print()
    print("   To view or use the key:")
    print("   - The plaintext key was shown only once during generation")
    print("   - If lost, run 'python manage_auth.py reset' to generate a new key")
    
    return 0


def reset():
    """Reset API key (delete and regenerate)."""
    if not API_KEY_FILE.exists():
        print("ℹ️  No existing API key found, generating new one...")
        return generate()
    
    print("⚠️  WARNING: This will invalidate your current API key!")
    print("   All clients will need to be updated with the new key.")
    print()
    response = input("Are you sure you want to continue? (yes/no): ").strip().lower()
    
    if response != "yes":
        print("❌ Reset cancelled")
        return 1
    
    # Delete existing key
    API_KEY_FILE.unlink()
    print(f"✅ Deleted old API key from {API_KEY_FILE}")
    print()
    
    # Generate new key
    return generate()


def main():
    """Main entry point for the authentication management CLI."""
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    
    command = sys.argv[1].lower()
    
    if command == "generate":
        return generate()
    elif command == "show":
        return show()
    elif command == "reset":
        return reset()
    else:
        print(f"❌ Unknown command: {command}")
        print(__doc__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
