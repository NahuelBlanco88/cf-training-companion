# CF Training Companion MCP prototype

An independent, authenticated MCP adapter to the existing REST API. The
deployed backend is unchanged. The draft also updates the root API with a
credential gate, guarded record corrections, duplicate protection for MCP
metcons, and read-only startup schema verification. The adapter never imports
`app.py` or calls the API at startup. No credentials belong in git.

## Current scope

- Thirteen MCP tools for finding workout sets, metcons, latest results, a
  training day, statistics, analytics, logging sets/metcons, and identifying and
  correcting individual records.
- Workout writes use `POST /workouts/bulk` **without** `force=true`, followed by
  `GET /workouts/verify` and an exact comparison of saved IDs and fields.
- MCP metcon writes require a server-side duplicate guard with a transaction
  lock on PostgreSQL, then read the returned ID. Different performances on
  the same day remain valid. Since there is no persisted request key, a lost
  response still requires inspecting records before retrying.
- Corrections use GET-by-ID and PATCH with a version check under a row lock.
  Only explicitly supplied fields change. The REST GET and PATCH routes must
  be deployed before correction tools can be used.
- The live `public.workout.reps` column is VARCHAR. The API maps it as text
  and returns both the interpreted integer `reps` and original `reps_raw`, so
  legacy expressions remain visible; omitted correction fields stay intact.
- No generic HTTP, arbitrary SQL, export, delete, `force`, or `undo_last` MCP
  tool exists. The old GPT's legacy full-record PUT and DELETE routes remain
  available under its distinct credential until redesigned; plugin deletion
  parity has not been implemented.

## Authentication and deployment blockers

The MCP endpoint requires OAuth bearer access tokens signed with RS256, an
exact issuer and audience, per-tool `training:read`, `training:write`, or
`training:edit` scope, and
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
   destination. The draft defaults to `CF_API_SCHEMA_INIT_MODE=verify`, which
   checks mapped columns without DDL and fails startup if they are missing.
   `legacy` startup DDL remains an explicit option for disposable local setups
   only. The owner supplied a live column listing matching all mapped workout
   and metcon columns, including the text `workout.reps` column. Keys, grants,
   live PostgreSQL behavior, and a fresh backup still need checking before
   a production deploy.
   Do not connect the MCP service to production during compatibility mode.
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
   cannot PUT/DELETE or bypass duplicate checks, and GPT still reads and logs. No synthetic workout
   should be inserted into production. A failed auth cutover warrants
   inspecting configuration and a reviewed compatibility rollback.
4. Decide which supported OAuth provider and account subject will be used.
   Do not send or check in token values, database passwords or private keys.
5. After enforcing backend authentication and configuring OAuth, deploy the
   separate MCP service. Test anonymous rejection, read scopes, write scopes,
   and edit scopes in an isolated environment. Register its real HTTPS URL in
   ChatGPT developer mode. Only then generate plugin `mcp.json` for that URL.
6. Review backup/PITR and deletion protection separately before changing
   production. Avoid running the repository's existing `test_app.py` against
   your logs; its fixture can drop tables.

In enforced mode the GPT token grants the existing reads, creates, edits and
deletes; MCP allows reads, guarded creates and conditional PATCH corrections
only; the optional read token allows
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
| `CF_API_MCP_TOKEN` | MCP reads, guarded creates, and conditional PATCH; secret store only |
| `CF_API_SCHEMA_INIT_MODE` | `verify` by default; do not use `legacy` on production tables |
| `CF_API_READ_TOKEN` | Optional read-only token; secret store only |

The bridge's `CF_API_TOKEN` must equal the backend's `CF_API_MCP_TOKEN`. Do
not put any tokens into a build file, GPT instructions, schema, test output,
or Git history. If the production dashboard calls this backend, identify its
configured target and give it a read-only credential before enforcement.

Only localhost HTTP URLs are accepted for offline development. Unset
configuration fails startup. The adapter does not print credential values.

## Plugin package

`plugins/cf-training-companion/` contains a portable plugin manifest and a
bundled training-log skill. It has **no `mcp.json` yet**, because no real MCP
HTTPS endpoint or OAuth provider has been configured. Do not replace that
unknown address with the existing REST URL. Once the bridge is deployed and
authenticated, generate the MCP config with the verified URL:

```sh
python -m mcp_bridge.render_plugin_config \
  --public-url https://ACTUAL-DEPLOYED-MCP-HOST/mcp \
  --output plugins/cf-training-companion/mcp.json
```

This generated file is ignored by Git until its target is verified and the
plugin is ready for installation. The existing GPT continues to use its
Action during the cutover.

## Local verification

From the repository root, in an isolated Python environment:

```sh
pip install -r mcp_bridge/requirements.txt fastapi sqlalchemy aiosqlite asyncpg pytest
pytest -q mcp_bridge/test_bridge.py mcp_bridge/test_api_auth.py mcp_bridge/test_isolated_api.py mcp_bridge/test_plugin_package.py
```

MCP tests use an in-process fake REST API and generated test tokens. The
isolated API test imports the root app only after selecting a fresh temporary
SQLite file, never enters its legacy startup initializer, and exercises only
that file. No test calls Cloud Run, Cloud SQL or the GPT Action; do not run
the repository's existing `test_app.py` against workout databases.
