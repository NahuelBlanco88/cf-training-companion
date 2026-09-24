# CF Training Companion MCP prototype

An independent, authenticated MCP adapter to the existing REST API. The
production `Dockerfile` at the repository root and the deployed backend are
unchanged. This adapter never imports `app.py` and cannot create or alter SQL
tables. It does not call the API on startup. No credentials belong in git.

## Current scope

- Nine MCP tools for finding workout sets, metcons, latest results, a training
  day, training statistics, analytics, and logging workout sets or one metcon.
- Workout writes use `POST /workouts/bulk` **without** `force=true`, followed by
  `GET /workouts/verify` and an exact comparison of saved IDs and fields.
- Metcon writes check existing results first, then use `POST /metcons` and a
  date/name read-back of the new ID. The current API has no dedicated metcon
  duplicate guard or GET-by-ID; read-back can therefore be incomplete. When a
  response is lost, writes are never retried automatically.
- No raw SQL, generic HTTP, export, edit, delete, or `undo_last` tool is exposed.
  Editing is blocked until the backend supports a safe read-by-ID and partial
  update that does not erase unrelated fields. This is **not** feature parity
  with the current GPT Action yet.

## Authentication and deployment blockers

The MCP endpoint requires OAuth bearer access tokens signed with RS256, an
exact issuer and audience, `training:read` and `training:write` scopes, and
one allowed OAuth subject. Configure an external OAuth 2.1 authorization
server with MCP discovery, PKCE and ChatGPT-compatible registration; this
adapter is a resource server and **does not implement login or issue tokens**.
The server publishes MCP protected resource metadata via the official SDK.

The backend API is **currently public and does not validate credentials**.
`CF_API_TOKEN` is mandatory to start the adapter, but a token header is not
protection until the backend rejects missing/incorrect tokens. Therefore:

1. Do not deploy or connect this prototype to production yet. Add and review
   authentication for the existing REST API while preserving the old GPT's
   logging path. Confirm unauthenticated and invalid-token calls are rejected.
2. Decide which supported OAuth provider and account subject will be used.
   Do not send or check in token values, database passwords or private keys.
3. Add read-by-ID and partial update routes to the backend and test on an
   isolated DB before enabling edit tools. Add server-side metcon idempotency.
4. Review backup/PITR and deletion protection separately before changing
   production. Avoid running the repository's existing `test_app.py` against
   your logs; its fixture can drop tables.

The current backend requires an API Bearer credential for this design. A
reviewed rollout must add enforcement before any production MCP connection.
The JSON Action schema does not itself contain the live GPT Authentication
setting; the owner confirmed it is currently `None`.

## Required environment variables

| Name | Purpose |
| --- | --- |
| `CF_API_URL` | Authenticated REST API root; no trailing path |
| `CF_API_TOKEN` | Backend Bearer credential; supply through a secret store |
| `MCP_PUBLIC_URL` | External HTTPS URL ending `/mcp` |
| `OAUTH_ISSUER` | Exact token `iss` value and authorization server URL |
| `OAUTH_AUDIENCE` | Exact token `aud`, same as `MCP_PUBLIC_URL` |
| `OAUTH_JWKS_URL` | Trusted issuer's HTTPS JWKS endpoint |
| `MCP_ALLOWED_SUBJECT` | Exact OAuth `sub` permitted to access this personal log |
| `PORT` | Optional HTTP port, default 8080 |

Only localhost HTTP URLs are accepted for offline development. Unset
configuration fails startup. The adapter does not print credential values.

## Local verification

From the repository root, in an isolated Python environment:

```sh
pip install -r mcp_bridge/requirements.txt pytest
pytest -q mcp_bridge/test_bridge.py
```

Tests use an in-process fake REST API and generated test tokens. They do not
import, start or call `app.py`, Cloud Run, Cloud SQL or the GPT Action.
