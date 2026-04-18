---
sidebar_position: 15
title: "AgentPhone"
description: "Set up Hermes Agent as a voice phone bot via AgentPhone"
---

# AgentPhone Setup

Hermes connects to [AgentPhone](https://agentphone.to) to place and answer real phone calls. AgentPhone transcribes the caller's speech and POSTs a webhook to Hermes; the HTTP response body is what gets spoken back to the caller via TTS.

Unlike text platforms, AgentPhone's conversational channel is the HTTP response body itself. Hermes streams its reply back as newline-delimited JSON (`application/x-ndjson`) so the caller starts hearing audio within milliseconds of the agent generating text, instead of waiting for the full turn to complete.

:::info No New Python Dependencies
The AgentPhone adapter uses `aiohttp` (already required by the webhook/api_server platforms) and `httpx` (core dependency). No additional packages are required.
:::

---

## Prerequisites

- An AgentPhone account at [agentphone.to](https://agentphone.to) with an agent created
- A publicly reachable HTTPS endpoint pointing at the Hermes gateway (so AgentPhone can deliver webhooks — use a reverse proxy, Cloudflare Tunnel, ngrok, etc.)

## Quick setup

Run the interactive wizard:

```bash
hermes gateway
```

Pick **AgentPhone** and follow the prompts. Alternatively, set the environment variables directly:

```bash
export AGENTPHONE_API_KEY=sk-...
export AGENTPHONE_AGENT_ID=agt_...
export AGENTPHONE_AGENT_PHONENUMBER=+15551234567
export AGENTPHONE_ALLOWED_INBOUND_NUMBERS=+15559876543,+15550000001
export AGENTPHONE_WEBHOOK_SECRET=whsec_...
# Optional:
# export AGENTPHONE_PORT=8646
# export AGENTPHONE_HOST=0.0.0.0
# export AGENTPHONE_BASE_URL=https://api.agentphone.to
```

Then configure the webhook in the AgentPhone dashboard to POST to:

```
https://<your-public-host>/agentphone/webhook
```

## Webhook direction: `from` vs `to`

Every AgentPhone webhook describes one turn of a live call, and which side of the `from` / `to` fields the human sits on depends on who placed the call:

| Direction | Placed by | `from` | `to` | Human (subject of the turn) |
|---|---|---|---|---|
| **inbound** | A person dialling in | The caller | The agent's number | `from` |
| **outbound** | The agent (via `send_message` / `send_message_tool`) | The agent's number | The person the agent dialled | `to` |

The adapter resolves direction from the payload's `direction` field when present and falls back to inferring it by comparing `from` against `AGENTPHONE_AGENT_PHONENUMBER`. The "other party" is then passed through as the session's `user_id` / `user_name`, used for per-caller intent lookup, and (on inbound only) checked against `AGENTPHONE_ALLOWED_INBOUND_NUMBERS`. Outbound webhooks are never allowlist-gated — the agent is allowed to dial any valid E.164 number.

## How calls are scoped

AgentPhone calls are different from chat messages: the recipient of a call is an unfamiliar person who may try to socially engineer the agent. Hermes applies four layers of defense:

1. **Session isolation.** Each call gets a brand-new Hermes session keyed on the AgentPhone call `id`. The agent on the call starts with no history from your other platforms.
2. **Restricted toolset.** During an inbound call Hermes loads a configurable tool list (default: `web_search`, `web_extract`, `todo`, `memory`, `session_search`). Tools not in this list — `send_message`, `cronjob`, `terminal`, filesystem, Home Assistant, browser, etc. — are not registered for the turn, so the caller cannot coax the agent into using them. Configure via `AGENTPHONE_CALL_ALLOWED_TOOLS` or `extra.call_allowed_tools` in config.yaml.
3. **Bound call intent.** Every outbound call requires a structured `intent` and `context_brief`. Those become the agent's only source of facts for the duration of the call; the agent is instructed to refuse anything outside that scope. For inbound calls the agent uses a conservative default intent configurable via `default_inbound_intent`.
4. **Prompt-injection scan + turn budget.** Every inbound transcript is scanned for known prompt-injection patterns (`ignore previous instructions`, invisible unicode, etc.) — a match short-circuits to `{"text": "I can't help with that. Goodbye.", "hangup": true}` without invoking the agent. A per-call turn budget (default 12) puts an upper bound on conversations that get past the scanner.

## Placing outbound calls

From the agent, use `send_message` with a structured intent:

```
send_message(
    target="agentphone:+15559876543",
    message="Hi, this is Kevin's assistant.",
    intent="Remind the user about their San Francisco trip on Apr 20",
    context_brief="Flight AA123 at 8am from SFO. Hotel: Marriott Union Square."
)
```

`intent` and `context_brief` are **required** for AgentPhone targets. Outbound calls can be placed to any valid E.164 number — there is no outbound allowlist.

## Security model (summary)

| Control | Enforced by |
|---|---|
| Inbound caller must be in allowlist | `AGENTPHONE_ALLOWED_INBOUND_NUMBERS` / `AGENTPHONE_AGENT_PHONENUMBER` |
| Outbound calls | Unrestricted — any valid E.164 number |
| Webhook authenticity | HMAC-SHA256 of `{timestamp}.{body}` with `AGENTPHONE_WEBHOOK_SECRET`, 5-minute replay window |
| Call stays on topic | Rigid `CALL PURPOSE` / `FACTS YOU MAY SHARE` / `FORBIDDEN TOPICS` system prompt |
| Minimum-surface toolset | Configurable per-call tool list (`AGENTPHONE_CALL_ALLOWED_TOOLS`) |
| Known jailbreak patterns | Transcript scan reuses `agent/prompt_builder._CONTEXT_THREAT_PATTERNS` |
| Runaway conversations | Per-call turn budget (default 12, configurable via `default_max_turns`) |

## Configuration reference

| Env var | Required | Purpose |
|---|---|---|
| `AGENTPHONE_API_KEY` | yes | Bearer token for `POST /v1/calls` |
| `AGENTPHONE_AGENT_ID` | yes | AgentPhone agent id that places/receives calls |
| `AGENTPHONE_AGENT_PHONENUMBER` | yes | The agent's own E.164 number (also implicitly allowed inbound) |
| `AGENTPHONE_ALLOWED_INBOUND_NUMBERS` | yes (for real use) | Comma-separated E.164 numbers; gates inbound callers only (outbound is unrestricted) |
| `AGENTPHONE_WEBHOOK_SECRET` | recommended | Webhook signing secret (starts with `whsec_`). If unset, signature verification is skipped with a warning. |
| `AGENTPHONE_PORT` | no (default `8646`) | Local port for the inbound webhook listener |
| `AGENTPHONE_HOST` | no (default `0.0.0.0`) | Bind host |
| `AGENTPHONE_BASE_URL` | no | Override the AgentPhone API base URL (for testing / on-prem) |
| `AGENTPHONE_CALL_ALLOWED_TOOLS` | no | Comma-separated list of tools the agent may use during an inbound call. Default: `web_search,web_extract,todo,memory,session_search`. Add tools to make it more permissive; remove tools to restrict further. Tools not in this list are not registered for the turn. |
| `AGENTPHONE_VOICE` | no | Default TTS voice for outbound calls (e.g. `Polly.Amy`, `Polly.Joanna`). Unset → AgentPhone's own default. Override per-call via `send_message(..., voice="Polly.Joanna")`. |
| `AGENTPHONE_MODEL` | no | Per-call model override (e.g. `anthropic/claude-haiku-4-5`). When set, every inbound webhook turn is processed by this model instead of the gateway default — useful for picking a fast latency-tuned model for voice while keeping a heavier default for chat. Provider/api_key/base_url are inherited from the gateway, so the model must be reachable through the same provider. Unset → use the gateway default. |

## Post-call summaries

When the agent places an outbound call from a Telegram / Slack / iMessage / etc. conversation, Hermes remembers where that prompt came from and delivers a post-call summary back to the same chat once the call ends.

The summary is built from AgentPhone's own `agent.call_ended` webhook — it includes the call's duration, `disconnectionReason` (agent vs. recipient hangup), `userSentiment`, `callSuccessful`, and the pre-computed `summary` field AgentPhone generates. No additional LLM call is required in the default flow.

Delivery is controlled by one extra key in `platforms.agentphone.extra`:

```yaml
platforms:
  agentphone:
    extra:
      summary_delivery: always        # always | only_answered | off
      stale_interaction_timeout_s: 600
```

- **`summary_delivery: always`** (default) — send a summary whenever an `agent.call_ended` event fires.
- **`summary_delivery: only_answered`** — skip summaries when `status != "completed"` (no-answer, busy, failed).
- **`summary_delivery: off`** — disable summaries entirely.

If AgentPhone ever fails to deliver the `agent.call_ended` event, a background reaper (default 10 min idle threshold, configurable via `stale_interaction_timeout_s`) will synthesise a summary from local state so the user isn't left hanging.

### Cross-process limitation

Origin tracking requires the `send_message` tool to run in-process with a live gateway adapter. When `hermes send-message` is invoked as a fresh subprocess (e.g. from an external cron or shell script) there is no adapter to register with, so the summary will not be delivered. In that case the tool's response includes a `note` field describing the fallback behaviour.
