<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ADR-014: Admin UI Testing & Coverage Strategy

**Status:** Accepted  
**Date:** 2025-12-07

## Context

The Admin UI is now the primary way operators manage the MCP server (status monitoring, index rebuilds, vector management, log tailing, configuration export). Recent regressions surfaced in the Vector/Logs/Config pages because tests were ad-hoc and coverage was <30%. Timer-driven UI (log auto-refresh) and dialog confirmation flows are prone to flaky behavior unless we enforce deterministic patterns in tests. We need consistent tooling, coverage goals, and guidelines so future changes don’t silently regress critical operational paths.

## Decision

1. Standardize on **Vitest + @testing-library/react** with jsdom for component tests.  
2. Require `npm run test -- --coverage` before merging Admin UI changes.  
3. Enforce the following coverage targets:
   - ≥70% overall statements/branches for the Admin UI package.
   - 100% line coverage for “critical flows”: `StatusPage`, `IndexPage`, `VectorPage`, `LogsPage`, `ConfigPage`, `src/utils/api.ts`, and shared dialog components used for rebuild/confirmation flows.
4. Timer-driven features must use deterministic patterns (store interval IDs, guard `window`/`navigator` access, expose `data-testid` for dialog confirmation buttons). Tests should avoid `setTimeout`/`setInterval` leakage by either mocking timers or stubbing `setInterval` return values.
5. Document the workflow (README + Runbook) so contributors and operators know how to run the suite and interpret coverage.

## Consequences

### Positive
- Prevents regressions in critical admin flows by making coverage expectations explicit.
- Encourages deterministic timer/async code, reducing flaky tests.
- Provides a repeatable “pre-release checklist” for Admin UI changes.

### Negative
- Slightly longer CI times due to coverage instrumentation.
- Contributors must invest time writing tests for new UI code.

### Neutral
- Does not dictate UI component libraries beyond current stack (React + shadcn/ui).

## Implementation Notes
- `vitest.config.ts` already configures jsdom + coverage via V8.
- `src/setupTests.ts` registers `@testing-library/jest-dom`.
- Shared utilities (e.g., `cn`, API client) must have tests hitting error paths to maintain 100% coverage.
- For clipboard or window APIs, wrap in guards (`if (typeof navigator !== 'undefined')`) so jsdom tests don’t crash.

## Related
- [docs/adr-011-admin-api.md](adr-011-admin-api.md) – Admin interface rationale.
- [README](../README.md) & [sigil-admin-ui/README.md](../sigil-admin-ui/README.md) – updated with testing instructions.
