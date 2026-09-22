# Setting up Claude Code on GitHub (self-hosted Volta runner)

This repo ships `.github/workflows/claude.yml`, which lets you drive Claude Code from
GitHub — comment `@claude <task>` on an issue/PR, or add the **`claude`** label to an
issue. Claude runs on a self-hosted runner **on your fleet** so it can
actually build/test/bench the CUDA kernels in Docker, then opens a PR.

There are three one-time steps (only you can do these — they need your browser, admin,
and hardware):

## 1. Install the Claude GitHub App

From the Claude Code CLI: `/install-github-app` (walks you through it), or install
manually from <https://github.com/apps/claude> and grant it this repo (Contents,
Issues, Pull Requests).

## 2. Add the auth secret (subscription token)

You're signed into Claude Code with a subscription, so mint a **long-lived** CI token
(the local cached credential is interactive-only and can't be refreshed by CI):

```bash
claude setup-token        # opens a browser for consent; prints a long-lived token
```

Then store it as a secret in each repo (never commit it):

```bash
# paste the token when prompted (gh reads it without echoing to logs)
for r in superl8 superl8-serve ComfyUI-superl8; do
  gh secret set CLAUDE_CODE_OAUTH_TOKEN -R jajmangold/$r
done
```

*(Alternative: a console.anthropic.com API key in `ANTHROPIC_API_KEY` — then flip the
commented lines in `claude.yml`.)*

## 3. Register a self-hosted runner on a fleet box

The runner must be a machine with the Volta/CMP GPUs + Docker + the NVIDIA container
toolkit (so `docker compose run --rm build|test|bench` works).

```bash
# GitHub → repo Settings → Actions → Runners → "New self-hosted runner" (Linux x64)
# gives you the exact download + ./config.sh command. When configuring, set labels:
./config.sh --url https://github.com/jajmangold/superl8 --token <RUNNER_TOKEN> \
            --labels self-hosted,volta-gpu,linux --name superl8-volta-1

# the runner user must be able to run docker + reach the GPUs:
sudo usermod -aG docker $(whoami)          # docker without sudo
# verify GPU visibility through Docker:
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi

# run it as a service so it survives reboots:
sudo ./svc.sh install && sudo ./svc.sh start
```

The workflow's `runs-on: [self-hosted, volta-gpu]` targets this runner. `docker
compose`'s `bench`/`profile` services already pin themselves to the profiling-capable
cards (host indices 4/7/9/11/14) per `CLAUDE.md`, so Claude honors the GPU rules.

One runner can serve all three repos if you register it at the **org/user level**
(Settings → Actions → Runners, add to a runner **group**), or register one per repo.

## ⚠️ Security — required for the public repos

`superl8-serve` and `ComfyUI-superl8` are public. A self-hosted runner on a public repo is a
real exposure: a stranger's PR could otherwise run code on your hardware. Two guards:

1. **Already in the workflow:** the `if:` condition only triggers for
   `OWNER`/`MEMBER`/`COLLABORATOR` authors — a random commenter can't start a run.
2. **You must also set:** repo Settings → Actions → General →
   *Fork pull request workflows from outside collaborators* → **Require approval for
   all external collaborators** (or "…for all outside collaborators"). This stops
   fork-PR code from touching the runner without your click.

Consider a dedicated, isolated box for the runner, and keep `CUDA_VISIBLE_DEVICES` /
the compose device pins off the counter-locked CMP cards for profiling.

## Using it

- **Mention:** comment `@claude fix the D=256 decode kernel and add a perf test` on any
  issue or PR. Claude replies, works on the runner, and opens/updates a PR.
- **Label:** apply the `claude` label to an issue to auto-start with no comment —
  good for "here's a scoped task, go."
- **Iterate:** comment again on the PR (`@claude also run trailmark`) to continue.

Claude auto-loads `CLAUDE.md` + `AGENTS.md`, so plan-mode-before-`csrc/`, the failing-
test-first TDD gate, "never edit `baseline.json` outside a baseline PR", and the
hardware truths all still apply in CI.

## Also available (no runner / no GPU)

- **Claude on the web** (<https://claude.ai/code> or `/web-setup`) runs tasks on
  Anthropic-managed VMs — great for the Python/serve/docs work, but **no GPU**, so not
  for kernel build/test.
- A **scheduled** workflow (cron + a `prompt:` input) can run recurring automation
  (e.g. nightly suite + open an issue on regressions).
