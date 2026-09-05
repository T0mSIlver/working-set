# Rules for agents working on this repo

- **Explorer layout:** stacked blocks never touch. Cards carry `margin-bottom:14px` inline, chart grids get 14 px from `.charts + .charts, .charts + .panel`, tiles blocks carry their own `margin:16px 0`. A new card or chart grid placed next to an existing block must get its 14 px; two cards with rounded corners touching is the tell. Check the seam in a screenshot before you call the change done.
- **Explorer verification:** a change is not verified by reading the diff. Serve `interactive/` over HTTP and drive the page in headless Chromium; a refactor must leave `frontierTable` text and every chart's SVG byte-identical.
