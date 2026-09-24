# Candidate OAuth setup: Auth0

This is a preparation checklist, not a record that Auth0 has been configured.
No Auth0 account, tenant, client, API, or production MCP endpoint has been
verified yet. The existing GPT Action keeps using the current API during this
work. Never paste tokens, passwords, signing keys, or client secrets into chat
or commit them to this repository.

## Identity provider setup

1. Create or select an Auth0 tenant that the owner controls. Note its issuer
   URL (including its exact trailing slash) and the registered API identifier.
   Configure an Auth0 API with identifier equal to the eventual *exact*
   `https://MCP-HOST/mcp` URL. Define `training:read`, `training:write`, and
   `training:edit` as API permissions. This URL cannot be finalized until the
   separate MCP service has an HTTPS hostname.
2. In tenant Settings > Advanced, confirm **Resource Parameter Compatibility
   Profile** is enabled. ChatGPT sends `resource` during authorization and token
   exchange. Auth0 must turn that into a signed access token whose `aud` equals
   the registered API identifier. Do not rely on a token for `/userinfo` or
   on an ID token as an MCP access token.
3. Choose *one* ChatGPT client registration path. For a personal integration,
   prefer a manually registered, dedicated third-party application if the
   ChatGPT plugin setup accepts its predefined client ID and the callback URL
   displayed in the connection manager. Otherwise DCR requires deliberately
   enabling Open Dynamic Registration, configuring default API permissions for
   third-party applications, and a domain-level login connection. Open DCR
   allows anyone to register a client in the tenant; review that exposure
   before enabling it. Do not place client credentials in the plugin manifest.
4. Enable authorization-code + PKCE (`S256`) and refresh tokens with
   `offline_access` for the selected client. Scope the API client grant to the
   three permissions above. Limit actual data access to the owner's exact
   verified Auth0 `sub` through `MCP_ALLOWED_SUBJECT`; email is not the `sub`.
   The bridge also verifies issuer, audience, signature, expiry, and scopes.
5. Read, but do not modify, the tenant's OIDC or OAuth discovery document and
   JWKS URL. Check that the discovery advertises `S256` and a compatible
   registration or configured-client path. Record only the public issuer URL,
   JWKS URL, API identifier, and account `sub` in private deployment settings.

## Offline and isolated validation before connecting personal records

- Confirm the MCP server advertises protected-resource metadata with its
  canonical `/mcp` URL and the Auth0 issuer; verify each tool declares its
  required training scope. The current SDK advertises an empty global scope
  list because scope requirements differ per tool.
- Sign in with a dedicated test account; check the access token's `iss`, `aud`,
  `sub`, and `scope` **locally without printing the token**. Require exact
  audience equality with the MCP URL. Reject wrong audience, subject, expired
  token, and missing scopes. A login screen alone does not establish that the
  token is valid for this server.
- Test read, write, and edit paths against isolated fixtures; verify failed
  writes are not retried silently, and no tool exposes delete, SQL, `force`,
  or unrestricted export. Run the offline pytest tests in `README.md`.
- Before any production rollout, separately review the backend auth cutover,
  secret storage, deployment commands, current backup, and the GPT Action
  API-key setting. Deploy and verify backend protection before exposing the
  MCP endpoint; do not disconnect the existing GPT Action prematurely.

Auth0 documentation: [resource-parameter setting](https://auth0.com/docs/get-started/tenant-settings),
[PKCE and resource parameter](https://auth0.com/docs/api/authentication/authorization-code-flow-with-pkce/authorize-with-pkce),
[dynamic client registration](https://auth0.com/docs/get-started/applications/dynamic-client-registration).
ChatGPT documentation: [plugin MCP authentication](https://developers.openai.com/plugins/build/auth).
