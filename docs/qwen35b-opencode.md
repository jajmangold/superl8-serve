# Qwen35B OpenCode option

The workspace `opencode.json` exposes the qualified four-replica Qwen3.6-35B-A3B
TQ3_4S fleet as `qwen35b-local/coder`. It is optional: the hosted Go provider and
the default model remain unchanged.

The proxy is local and does not require an API key. Set the endpoint and a stable,
per-code-session affinity value before starting an OpenCode process:

```sh
export QWEN35B_PROXY_BASE_URL=http://127.0.0.1:8017/v1
export QWEN35B_CODE_SESSION="opencode-$$"
opencode run --model qwen35b-local/coder "your coding task"
```

`QWEN35B_CODE_SESSION` is sent as `X-Code-Session` and is hashed by the Nginx
front door, keeping that process's requests on one llama-server replica. Use a
different value for each independent code session to spread sessions across the
four CMP cards. Do not put credentials or model weights in this file.

The model advertises a 260,000-token context and an 8,192-token output limit. The
fleet's container and proxy deployment is defined in `docker/qwen35-tq3/`.
