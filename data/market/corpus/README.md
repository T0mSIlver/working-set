# data/market/corpus/

Fixed text that `scripts/market_tokenizers.py` tokenizes with every catalogue
model's tokenizer, to turn prices per token into prices per character. The
files are frozen copies: editing them changes every number in
`data/market/chars_per_token.json`, whose `corpus` block records each kind's
character count, byte count and SHA-256.

| Kind | File | Chars | Source | Licence |
|---|---|---|---|---|
| `code` | `code/shared.py` | 29,965 | first 30 kB (whole lines) of `src/workingset/shared.py` at commit `e554722` | MIT, this repo (`LICENSE`) |
| `code` | `code/charts.js` | 25,889 | first 26 kB (whole lines) of `interactive/src/charts.js` at commit `e554722` | MIT, this repo (`LICENSE`) |
| `english` | `english/verne_around_the_world_ch1-4.txt` | 32,575 | Jules Verne, *Around the World in Eighty Days*, chapters I–IV, Project Gutenberg eBook #103 (`103-0.txt`, lines 56–765) | public domain |
| `french` | `french/verne_tour_du_monde_ch1-4.txt` | 37,710 | Jules Verne, *Le Tour du monde en quatre-vingts jours* (1873), chapters I–IV, Project Gutenberg eBook #800 (`800-8.txt`, ISO-8859-1 converted to UTF-8, lines 143–971) | public domain |
| `json_tool_calls` | `json_tool_calls/chat_completions_request.json` | 32,277 | composed for this corpus, see below | MIT, this repo |

## Prose

English and French are the same four chapters of the same novel, so the two
kinds carry the same content. Verne died in 1905 and the English translation
dates from the 1870s; both are public domain. Project Gutenberg's licence says
of its public-domain eBooks: "You may use this eBook for nearly any purpose
such as creation of derivative works, reports, performances and research. [...]
Redistribution is subject to the trademark license". The files here keep only
the chapter text: the Gutenberg header, footer and name are cut, so the
trademark licence does not apply. Line endings are LF.

## Code

Python and JavaScript from this repository, copied rather than linked so later
edits to the originals do not move the numbers.

## JSON tool calls

One OpenAI chat-completions request body, serialised compactly as an SDK sends
it: four tool schemas a coding agent uses (`read_file`, `edit_file`, `bash`,
`grep`), then a conversation of assistant `tool_calls` (arguments as JSON
strings) and `tool` results. It was composed for this corpus: the user prompt,
the tool calls and the final answer were written by hand, and every tool result
is the real output of that call run over this repository at commit `e554722`
(file excerpts from `src/workingset/model.py`, `src/workingset/data/catalog.toml`,
`interactive/src/config.js`, `tests/test_model.py`, grep hits, `wc -l`), except
the `edit_file` and `git diff --stat` results, which describe the hand-written
edit. It contains nothing from a real agent session.
