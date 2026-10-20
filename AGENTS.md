# Rules for agents working on this repo

- **Explorer layout:** stacked blocks never touch. Cards carry `margin-bottom:14px` inline, chart grids get 14 px from `.charts + .charts, .charts + .panel`, tiles blocks carry their own `margin:16px 0`. A new card or chart grid placed next to an existing block must get its 14 px; two cards with rounded corners touching is the tell. Check the seam in a screenshot before you call the change done.
- **Explorer verification:** a change is not verified by reading the diff. Serve `interactive/` over HTTP and drive the page in headless Chromium; a refactor must leave `frontierTable` text and every chart's SVG byte-identical.
- **The model and its mirror:** `src/workingset/model.py` is the source of truth; `interactive/src/*.js` mirrors it. A modelling change lands in Python FIRST, then in the JS. `tests/golden/` pins the two together — regenerate with `uv run scripts/golden.py`, check with `node --import ./tests/js/register.mjs --test tests/js/golden.test.mjs`. That test going red after a model change is the intended signal, not the bug; see `tests/golden/README.md` before touching `known_disagreements.json`.

## Fast lane (since 2026-10-20)

Nobody outside uses working-set yet, so changes go straight to `main`: commit, run
the fast local tests, push. CI runs on `main`; fix a red run forward before new
work. There is no per-change cross-review: a nightly job (`fastlane-review`)
reviews the day's commits on `main` and writes findings to
`~/work/working-set/.scratch/daily-review/<date>.md`. Fix the P0 and P1 findings in
any file there before new work, then delete the file.

These still go through a PR, a cross-review and a card to Tom: `interactive/**` and
`wrangler.jsonc` (a push there deploys workingset.tomvaucourt.com), `.github/`,
release tags (`v*` publishes to PyPI), and anything else that publishes.

Everything goes back through PRs at the first outside user holding data, the
first outside PR, or the first build in a channel other people install from.
