<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ADR-004: JSON Configuration with Environment Variable Fallback

## Status

Accepted

## Context

The Sigil MCP server needs configuration for:

- Repository paths (which repos to index)
- Server settings (host, port, name)
- Authentication settings (OAuth, API keys, local bypass)
- Index storage location
- Logging levels

Initial implementation used environment variables exclusively:
```bash
export SIGIL_REPO_MAP="name:/path;other:/other/path"
export SIGIL_MCP_AUTH_ENABLED=true
export SIGIL_MCP_OAUTH_ENABLED=true
```

This worked but had limitations:

- **Complex syntax**: Semicolon-separated repo map is error-prone
- **Poor discoverability**: Users don't know what variables exist
- **No validation**: Typos in variable names silently ignored
- **Hard to share**: Environment variables not version-controlled
- **Platform-specific**: Different shells handle exports differently

For a public GitHub repository, we need a more user-friendly configuration approach.

## Decision

Implement JSON configuration file with environment variable fallback:

1. **Primary: config.json**: JSON file with structured configuration
2. **Fallback: Environment Variables**: Backward compatible with existing deployments
3. **Example Template**: `config.example.json` showing all options
4. **Config Class**: Python class with property accessors and validation
5. **Search Path**: Look in current directory, parent directory, home directory
6. **Git Ignore**: Exclude `config.json` (contains paths, potentially sensitive)

Configuration precedence:
1. Environment variables (highest priority)
2. `./config.json` (current directory)
3. `../config.json` (parent directory)
4. `~/.sigil_mcp_server/config.json` (home directory)
5. Defaults (lowest priority)

Example config.json:
```json
{
  "server": {
    "name": "sigil_repos",
    "host": "127.0.0.1",
    "port": 8000
  },
  "repositories": {
    "my_project": "/absolute/path/to/project"
  }
}
```

## Consequences

### Positive

- **User-friendly**: JSON is familiar, widely supported, easy to edit
- **Discoverable**: `config.example.json` documents all available options
- **Validatable**: JSON schema can validate structure before startup
- **Version-controllable**: Teams can share example configs in repos
- **IDE support**: Editors provide autocomplete and validation for JSON
- **Backward compatible**: Existing environment variable setups still work
- **Flexible**: Can use config file for defaults, env vars for overrides

### Negative

- **Another file to manage**: Users need to copy and edit config.example.json
- **Path management**: Config file lookup across multiple directories adds complexity
- **Merge conflicts**: If config.json accidentally committed, team conflicts possible
- **Security risk**: If committed with secrets (mitigated by .gitignore)

### Neutral

- Config class at `config.py` handles loading and precedence
- Both JSON and environment variable formats supported indefinitely
- Config validation happens at server startup (fail fast)
- Repository paths resolved and validated (must exist as directories)

## Alternatives Considered

### Alternative 1: YAML Configuration

Use YAML instead of JSON for configuration.

**Rejected because:**
- Requires additional dependency (PyYAML)
- JSON is Python stdlib (no external deps)
- YAML's whitespace sensitivity can cause subtle errors
- JSON is simpler and more widely supported
- For this use case, JSON's limitations don't matter (no multi-line strings needed)

### Alternative 2: TOML Configuration

Use TOML (like Rust's Cargo.toml).

**Rejected because:**
- Less familiar to general developers than JSON
- Requires external dependency (tomli/tomllib)
- JSON's structure is sufficient for our needs
- Not as widely supported by editors

### Alternative 3: Python Configuration File

Execute Python file as configuration (Django-style).

**Rejected because:**
- Security risk (executing arbitrary code)
- Harder for non-Python users to edit
- No schema validation
- Poor fit for simple key-value configuration
- Potential for hard-to-debug errors

### Alternative 4: Environment Variables Only

Keep environment variables as the only configuration method.

**Rejected because:**
- Poor user experience for public repository
- Complex syntax for repository maps (semicolon-separated)
- Hard to discover available options
- Not version-controllable
- Doesn't scale to more complex configuration needs

### Alternative 5: Command-Line Arguments

Use CLI arguments for all configuration.

**Rejected because:**
- Very long command lines for multiple repos
- Hard to persist settings
- Poor fit for configuration that rarely changes
- Would still need config file for complex settings
- Startup scripts become unwieldy

## Related

- `config.py` - Configuration loader implementation
- `config.example.json` - Example configuration template
- `.gitignore` - Excludes config.json from version control
- [README Configuration Section](../README.md#configuration)
