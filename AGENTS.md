# Rules for agents working on this repo

- **Explorer layout:** every stacked block (chart grid, card, tiles) is 14 px from the next — cards carry `margin-bottom:14px` inline, chart grids get it from `.charts + .charts, .charts + .panel`. A new block placed next to an existing one must have it; two cards with rounded corners touching is the tell. Check the seam in a screenshot before you call the change done.
- **Explorer verification:** a change is not verified by reading the diff. Serve `interactive/` over HTTP and drive the page in headless Chromium; a refactor must leave `frontierTable` text and every chart's SVG byte-identical.
