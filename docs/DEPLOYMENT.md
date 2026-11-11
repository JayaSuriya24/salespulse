# Deployment guide

## Selected free platform: Render

Hugging Face Docker Spaces is not selected because the current Docker Spaces compute tier requires a paid plan for personal accounts. Render documents a free web-service tier suitable for demos, with important limits.

1. Push the project to a GitHub repository.
2. Open Render and create a new Blueprint from the GitHub repository.
3. Render reads `render.yaml`; confirm the service plan remains `free`.
4. Wait for the build and deployment to finish.
5. Open the assigned `onrender.com` URL.
6. Check `/api/v1/health` and `/docs`.
7. In the Render dashboard open the service > **Environment**, and copy the value of `M5_UPLOAD_TOKEN` (generated automatically by `render.yaml`). Upload the three official M5 CSVs using the dashboard's M5 uploader, pasting that token into the "Upload token" field.
8. Confirm the model report is populated and generate a 7-day forecast.

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `PYTHON_VERSION` | set in `render.yaml` (3.11.11) | Pins a Python version supported by the pinned pandas |
| `M5_UPLOAD_TOKEN` | **yes, to use M5 upload** | Bearer token for `POST /api/v1/datasets/m5/upload`; `render.yaml` uses `generateValue: true` so Render creates it and nothing secret is committed. Without a valid value (16+ characters) the endpoint answers 503. |
| `PORT` | set by Render | Listen port used by the start command |
| `DATA_DIR`, `ARTIFACT_DIR`, `RAW_DIR` | optional | Override storage locations |
| `MAX_UPLOAD_MB` | optional (default 50) | Size cap for the generic CSV preview upload |

Rotate the token by editing `M5_UPLOAD_TOKEN` in the Render dashboard (it is read on every request, but Render restarts the service when environment variables change). Anyone with access to the Render service can read the value.

## Free tier limitations
- 0.1 CPU and 512 MB RAM.
- Spins down after 15 minutes without inbound traffic.
- Local filesystem is ephemeral; files may be lost after restart or sleep.
- The service may be restarted by the platform.
- No persistent disk is available on the free web-service plan.

Do not add a payment method or select a paid plan for this project. If free-tier limits prevent the full dataset workflow, use a smaller subset and document it rather than upgrading.
