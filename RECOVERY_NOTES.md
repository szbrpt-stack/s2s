# S2S Recovery Notes

This file documents the safe recovery reference for the current integrity/staging work.

## Production reference

- Repository: `szbrpt-stack/s2s`
- Production branch: `main`
- Known stable production commit before integrity work: `889f06350373b698d97b16c03b3d8721fa1add6e`
- Production Render service: `s2s-1`
- Production URL: `https://s2s-1-na3v.onrender.com`

The integrity work must not be promoted to production until staging validation is complete.

## Staging reference

- Branch: `fix/runtime-recovery-data-integrity`
- Render service: `s2s-integrity-staging`
- Staging URL: `https://s2s-integrity-staging.onrender.com`
- Staging uses PostgreSQL/Supabase through the configured Session Pooler `DATABASE_URL`.
- Do not commit database credentials or connection secrets to GitHub.

## Recovery if staging changes fail

1. Do not modify production as part of staging recovery.
2. Compare the staging branch against `main` before reverting anything.
3. The production reference commit above is the known rollback point for backend code prior to the integrity branch.
4. If a staging deploy fails, restore the staging branch/service to the last known-good staging commit rather than changing production.
5. Validate `/health`, `/api/v1/runtime/integrity`, `/api/v1/model/integrity`, and `/api/v1/model/empirical-integrity` after recovery.
6. Confirm PostgreSQL persistence initializes successfully and no calibration jobs remain incorrectly marked `RUNNING`.

## Mobile application

The mobile application source is currently local and is not present in this repository. Do not infer or recreate the mobile project from the backend repository. When the mobile source is available, connect it separately and use a staging backend configuration without overwriting the production application.

## Security note

Supabase currently reports RLS disabled on the public application tables. Do not enable RLS blindly until the application's access path and required database roles are verified, because an incorrect policy can interrupt backend access.
