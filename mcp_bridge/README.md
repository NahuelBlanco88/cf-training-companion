# CF Training Companion MCP prototype

An independent, authenticated MCP adapter to the existing REST API. The
deployed backend is unchanged. The root Dockerfile also includes the proposed
standalone API authentication module. This adapter never imports `app.py` and cannot create or alter SQL
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

The deployed backend API is **currently public and does not validate credentials**.
The draft includes a staged API auth gate in `api_auth.py`, `app.py`, and the
root Dockerfile. Its default `CF_API_AUTH_MODE=compat` preserves the current
GPT Action and leaves the API public. `CF_API_AUTH_MODE=enforce` checks
credentials. `CF_API_TOKEN` is mandatory to start the adapter, but a token
header is not protection until the deployed backend enforces authentication.
The reviewed cutover sequence is:

1. Before any deployment, verify a recent recoverable backup and the Cloud SQL
   destination. The root app still runs schema initialization on startup;
   deployment requires a separate production decision. Do not connect the
   prototype MCP service to production during compatibility mode.
2. Supply distinct random GPT and MCP Bearer tokens (at least 32 characters)
   from a secret store. An optional third token is read-only. Run the revised
   backend in `compat` while the old GPT Action sends no credential. Set the
   GPT Action API key in its private editor settings, then check a normal read
   and a real user-directed log while compatibility remains active. Official
   OpenAI documentation confirms GPT Action API key support; check the
   editor's actual Authorization header behavior before enforcing. Never put
   tokens in its JSON schema.
3. With the old GPT confirmed working with its credential, change
   `CF_API_AUTH_MODE=enforce` with both GPT and MCP tokens configured. Verify
   missing/invalid tokens return 401, read-only credentials cannot write, MCP
   cannot edit/delete, and GPT still reads and logs. No synthetic workout
   should be inserted into production. A failed auth cutover warrants
   inspecting configuration and a reviewed compatibility rollback.
4. Decide which supported OAuth provider and account subject will be used.
   Do not send or check in token values, database passwords or private keys.
5. Add read-by-ID and partial update routes to the backend and test on an
   isolated DB before enabling edit tools. Add server-side metcon idempotency.
6. Review backup/PITR and deletion protection separately before changing
   production. Avoid running the repository's existing `test_app.py` against
   your logs; its fixture can drop tables.

In enforced mode the GPT token grants the existing reads, creates, edits and
deletes; MCP allows reads and creates only; the optional read token allows
reads (including POST verification). `/health`, `/`, and CORS preflight remain
public. Unknown write paths are denied. The GPT edit/delete permission is a
legacy compatibility role, not a recommended permanent permission.

The proposed backend requires an API Bearer credential for this design. A
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

Backend-only settings, separate from the MCP service:

| Name | Purpose |
| --- | --- |
| `CF_API_AUTH_MODE` | `compat` by default; `enforce` only after GPT Action key works |
| `CF_API_GPT_TOKEN` | Existing GPT's legacy rights; secret store only |
| `CF_API_MCP_TOKEN` | MCP reads and creates; secret store only |
| `CF_API_READ_TOKEN` | Optional read-only token; secret store only |

The bridge's `CF_API_TOKEN` must equal the backend's `CF_API_MCP_TOKEN`. Do
not put any tokens into a build file, GPT instructions, schema, test output,
or Git history. If the production dashboard calls this backend, identify its
configured target and give it a read-only credential before enforcement.

Only localhost HTTP URLs are accepted for offline development. Unset
configuration fails startup. The adapter does not print credential values.

## Local verification

From the repository root, in an isolated Python environment:

```sh
pip install -r mcp_bridge/requirements.txt pytest
pytest -q mcp_bridge/test_bridge.py
pytest -q mcp_bridge/test_api_auth.py
```

Tests use an in-process fake REST API and generated test tokens. They do not
import, start or call `app.py`, Cloud Run, Cloud SQL or the GPT Action.
