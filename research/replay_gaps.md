# What the waits in recorded agent sessions are (2026-10-20)

`ws test --replay` loads an endpoint with recorded pi sessions, so the "comfortable
capacity" of [docs/scenarios.md](../docs/scenarios.md#planning-stance-owner-decisions-2026-07-21)
(the largest replayed population meeting the latency targets) can be measured
rather than modelled. A replay is only as good as its waits: a session left
open overnight replayed as a 28-hour pause is a seat that sends nothing. So
`ws trace pi` gives every wait a kind read from the session's events, not from
its length. This note counts the kinds in the study's own sessions and records
how a replay treats each.

## The corpus

Tom's pi sessions on the dev box, July to October 2026: 42 files with a request,
623 requests. Six of them ran under `/tmp` (a Claude session driving pi to test
it), so the counts below exclude them (`--exclude '*/--tmp*'`): **36 sessions,
603 requests**, 8.4 requests per message the person typed. Prompts are p50
42,470 tokens, p90 83,745. Replies are p50 187, p90 847. Only these counts
leave the files. The trace and the sessions stay on the box.

## Gap kinds

| kind | what the events show | n | p50 | p90 | max | total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tool | tools ran, nobody was asked anything | 457 | 0.0 s | 0.2 s | 31 s | 0.03 h |
| permission | a tool waited on a permission prompt (pi-permission-system's log: prompt shown → answered, by tool call id) | 70 | 3.7 s | 198 s | 822 s | 1.36 h |
| ask | the agent asked the person a question (`ask_user_question`) | 1 | 152 s | | | 0.04 h |
| think | the agent finished; the person read and typed | 34 | 15.3 s | 208 s | 1,215 s | 0.78 h |
| abort | the person interrupted a request and typed again (timed from the interruption) | 1 | 9.4 s | | | |
| error | a request failed (timeout, server down) and was retried | 3 | 4.3 s | 7.1 s | 7 s | |
| idle | a person's wait past 30 min with no restart in the file | 1 | 28.1 h | | | 28.1 h |
| resume | pi restarted on the file (an extension rewrote its start-up state) | 0 | | | | |
| branch | the person moved to another point of the session tree | 0 | | | | |
| compaction | pi compacted the history | 0 | | | | |

Not replayed, because the server's work on them is unknown: 4 interrupted and
11 failed requests. The six `/tmp` sessions add the only resumes (4, each 12 to
25 s: a script restarting pi per prompt). Compaction is off in this pi
configuration, so the corpus cannot show it. The classifier handles it for
corpora where it is on.

What this changes:

- **Tool time is almost nothing.** Without the permission log, the long "tool"
  waits (up to 14 min) looked like builds. They were a person answering a prompt.
  90% of tool gaps end within 0.2 s and 97% within 1 s, and 4,868 s of the
  4,886 s in permission gaps was the prompt standing. A deployment that auto-approves loses those
  waits, and its sessions get burstier.
- **Only one wait in the corpus needs a threshold.** The overnight one (28 h)
  has no event in the file around it. Everything else is classified by what
  happened in it.

## How a replay treats each kind

- **Waits:** every kind is replayed as recorded, capped at
  `--replay-max-gap-s` (1800 s), except the three where the session sat
  unused: error, resume and idle. `--replay-gap-policy` decides those:
  `sample` (default) replaces them with a think gap drawn from the same trace,
  `cap` keeps them capped, `drop` makes them 0 s. `sample` keeps the seat an
  active user. A capped 30-minute wait would keep a seat silent for a whole
  2-minute rung, and a dropped one would add a burst no person makes. Tom chose
  `sample` on 2026-10-20.
- **Request shape:** synthetic text of the recorded token counts. Each request
  is the previous prompt, the server's own reply and fresh text up to the
  recorded size, so the prefix cache hits and misses where it did. A prompt
  that shrank (compaction, a branch) restarts from the shared prefix, cold
  past it. Replaying the real prompts byte for byte would put session content
  (secrets included) on the wire and in the trace. It would also be cut by the
  recording model's tokenizer, not the target's.
- **Output length:** `max_tokens` is the recorded reply length, and
  `ignore_eos` makes the server generate all of it.
- **N users from 36 sessions:** each simulated session starts at a random
  request of a random recording, weighted by length, so a short window sees
  sessions at every stage. When its recording ends it moves on to another one.
  Seats that replay the same recording share only the system prefix, since
  their text is drawn per seat.

## End to end

```bash
uv run ws trace pi --exclude '*/--tmp*' -o trace.json
uv run ws test workingset.toml --exclusive --replay trace.json --dry-run
```

For the 27B on 4×H200 the dry run plans rungs from 42 to 334 users. At the
recorded server's speed (12.4 s per request on average, then a 14.6 s mean
wait) 42 users offer 1.55 requests/s and 334 offer 12.4. A real run against a
local stub (2 rungs, 8 and 17 users, 50 s each) replayed 252 requests with no
error and printed the measured-capacity bracket. The capacity number itself
needs a run against the real endpoint.

Ties in with #89 (tool-result sizes and the reasoning/answer split: the trace
has sizes but not the split), #90 (a turn as a burst then a human gap: 8.4
requests per message here, with tool gaps near zero) and #102 (size
distributions: the trace keeps every request's size, so a replay already plays
the distribution).
