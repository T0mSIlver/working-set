import { CONFIG, makeGrid } from './config.js';
import { hasHeadcount, peopleFromSessions, state } from './state.js';
import { cssv, esc, fmt, linScale, logScale, logTicks, svgEl } from './svg.js';
import { PLANNER_COLORS, PLANNER_LABEL } from './planner.js';

/* ---- The frontier table ---- */
/* A DECISION table. Every column is one entry in frontierColumns(): its
   header, its cell and the value it sorts by. The default view is the verdict
   columns only (score on the active benchmark, the verdict at YOUR load, max
   users, what binds, €/seat); the rest are opt-in from the column picker.
   Clicking a header sorts by it; sort and optional columns are view state
   and re-render from the cached rows. The four ceilings ride state.showCeil
   so old share links that set it still open with them. */
export let lastFrontierRows = null, lastFrontierCurKey = null;
const OPTIONAL = { otherBench: false, headroom: false, bstar: false, eur: false };
let sortBy = null, sortDir = 1;   // null = the order the rows were handed in (max users)
function frontierColumns(){
  const C = PLANNER_COLORS();
  // status colors as var() references, so they track a theme flip without a
  // re-render (the bind column's chart palette still needs the redraw hook)
  const good='var(--good)', warn='var(--warn)', crit='var(--crit)', muted='var(--muted)';
  const viable = r => r.op.limit >= 1;
  const fits = r => viable(r) && r.op.fits;
  const room = r => viable(r) ? r.op.limit/r.op.users : NaN;
  // a censored decode search is a floor, not an estimate — carry the '≥'
  // through every figure derived from it, not just the headline
  const cen = r => r.op.binding==='decode' && r.censored ? '≥ ' : '';
  const dim = on => on ? '' : ` style="color:${muted}"`;
  const cols = [{ key:'name', head:'configuration', asc:true, sort:r => frontierRowName(r),
                  cell:r => `<td>${esc(frontierRowName(r))}</td>` }];
  // the model's scores, not the row's: every split of a model shares them.
  // The active benchmark is always shown; the other one is opt-in.
  for (const [k, b] of Object.entries(CONFIG.BENCHES)){
    if (k !== state.bench && !OPTIONAL.otherBench) continue;
    cols.push({ key:'bench:'+k, head:b.label, num:true, sort:r => frontierScore(r, k),
      cell:r => { const q = frontierScore(r, k);
        return `<td class="num"${k===state.bench?'':dim(false)}>${isFinite(q) ? fmt(q*100,1)+'%' : '—'}</td>`; } });
  }
  cols.push({ key:'load', head:'your load', sort:room,
    cell:r => viable(r)
      ? `<td class="v" style="color:${fits(r)?good:crit}">${fits(r)?'✓ fits':'✗ over'}</td>`
      : `<td class="v" style="color:${muted}">—</td>` });
  cols.push({ key:'users', head:hasHeadcount()?'max sessions / people':'max users', num:true,
    sort:r => viable(r) ? r.op.limit : NaN,
    cell:r => {
      if (!viable(r)){
        const why = r.op.binding === 'latency' ? 'cannot meet the TTFT budget at any load'
          : r.op.binding === 'saturation' ? 'prefill saturates before one user'
          : `${PLANNER_LABEL[r.op.binding]} allows under one user`;
        return `<td class="num" style="color:${muted}">not viable<br><span style="font-size:10.5px">${esc(why)}</span></td>`;
      }
      const c = cen(r);
      return hasHeadcount()
        ? `<td class="num">${c}${fmt(r.op.limit,0)} sessions<br><span style="color:${muted}">≈ ${c}${fmt(peopleFromSessions(r.op.limit),0)} people${r.reps>1?` · ${c}${fmt(r.op.limit/r.reps,0)}/grp`:''}</span></td>`
        : `<td class="num">${c}${fmt(r.op.limit,0)}${r.reps>1?` <span style="color:${muted}">(${c}${fmt(r.op.limit/r.reps,0)}/grp)</span>`:''}</td>`;
    } });
  cols.push({ key:'binds', head:'binds on', asc:true, sort:r => PLANNER_LABEL[r.op.binding],
    cell:r => `<td class="bind" style="color:${C[r.op.binding]}">${esc(PLANNER_LABEL[r.op.binding])}</td>` });
  if (OPTIONAL.headroom) cols.push({ key:'headroom', head:'headroom', num:true, sort:room,
    cell:r => { const x = room(r);
      return viable(r)
        ? `<td class="num" style="color:${fits(r)?(x>1.25?good:warn):crit}">×${fmt(x, x<10?1:0)}${cen(r)?'+':''}</td>`
        : `<td class="num" style="color:${muted}">—</td>`; } });
  if (state.showCeil) for (const k of ['cache','decode','latency','saturation'])
    cols.push({ key:'ceil:'+k, head:k, num:true, sort:r => isFinite(r.op.ceilings[k]) ? r.op.ceilings[k] : NaN,
      cell:r => `<td class="num"${k===r.op.binding?` style="color:${C[k]};font-weight:650"`:''}>`
        + `${k==='decode'&&r.censored?'≥ ':''}`
        + `${isFinite(r.op.ceilings[k])
            ? (hasHeadcount()
              ? `${fmt(r.op.ceilings[k],0)} sessions<br><span style="color:${muted}">≈ ${fmt(peopleFromSessions(r.op.ceilings[k]),0)} people</span>`
              : fmt(r.op.ceilings[k],0))
            : '—'}${k==='decode'&&r.capped?` <span style="color:${muted}">cap</span>`:''}</td>` });
  if (OPTIONAL.bstar) cols.push({ key:'bstar', head:'B*', num:true, sort:r => r.bstar,
    cell:r => `<td class="num">${fmt(r.bstar,1)}</td>` });
  // energy is meaningful only where the load can actually be served
  if (OPTIONAL.eur) cols.push({ key:'eur', head:'€/mo', num:true, asc:true, sort:r => viable(r) ? r.eur : NaN,
    cell:r => `<td class="num"${dim(fits(r))}>${viable(r) ? fmt(r.eur,0) : '—'}</td>` });
  // chart H's y: the bill with the row full, per user it then carries;
  // priced at a censored limit it is an upper bound, hence '≤'
  cols.push({ key:'seat', head:'€/seat', num:true, asc:true,
    sort:r => viable(r) && isFinite(r.eurSeat) ? r.eurSeat : NaN,
    cell:r => `<td class="num"${dim(fits(r))}>${viable(r) && isFinite(r.eurSeat) ? (cen(r)?'≤ ':'')+fmt(r.eurSeat,0) : '—'}</td>` });
  return cols;
}
export function renderFrontierTable(rows, curKey){
  lastFrontierRows = rows; lastFrontierCurKey = curKey;
  const cols = frontierColumns();
  const sc = cols.find(c => c.key === sortBy);
  // rows without a value (not viable, unscored) sink in either direction
  const sorted = sc ? rows.map((r, i) => [r, sc.sort(r), i]).sort(([, a, i], [, b, j]) => {
    const na = typeof a === 'number' && !isFinite(a), nb = typeof b === 'number' && !isFinite(b);
    if (na || nb) return na - nb || i - j;
    return (typeof a === 'string' ? a.localeCompare(b) : a - b) * sortDir || i - j;
  }).map(([r]) => r) : rows;
  const other = Object.entries(CONFIG.BENCHES).find(([k]) => k !== state.bench);
  const picks = [['otherBench', other ? other[1].label : 'other benchmark'], ['headroom', 'headroom'],
                 ['ceil', 'the four ceilings'], ['bstar', 'B*'], ['eur', '€/mo']];
  const picker = `<div class="colpick"><span class="lbl">columns</span>`
    + picks.map(([k, l]) => `<button type="button" data-col="${k}" aria-pressed="${k === 'ceil' ? state.showCeil : OPTIONAL[k]}">${esc(l)}</button>`).join('')
    + `</div>`;
  const head = '<tr>' + cols.map(c => {
    const on = c.key === sortBy;
    const arrow = on ? (sortDir > 0 ? ' ▲' : ' ▼') : '';
    return `<th${c.num ? ' class="num"' : ''} data-sort="${c.key}" aria-sort="${on ? (sortDir > 0 ? 'ascending' : 'descending') : 'none'}"`
         + `${c.key === 'bench:'+state.bench ? ' style="color:var(--text)"' : ''}>${esc(c.head)}${arrow}</th>`;
  }).join('') + '</tr>';
  const body = sorted.map(r => `<tr${r.key===curKey ? ' class="you"' : ''}>${cols.map(c => c.cell(r)).join('')}</tr>`).join('');
  document.getElementById('frontierTable').innerHTML =
    `${picker}<div class="ftable-wrap"><table class="ftable">${head}${body}</table></div>`;
}
// one listener for the header sort and the column picker; both re-render
// from the cached rows (display only, no recompute)
export function wireFrontierTable(){
  document.getElementById('frontierTable').addEventListener('click', e => {
    const th = e.target.closest('th[data-sort]'), pick = e.target.closest('button[data-col]');
    if (th){
      const c = frontierColumns().find(c => c.key === th.dataset.sort);
      if (sortBy === c.key){
        // third click on a column returns to the default order
        if (sortDir === (c.asc ? -1 : 1)) sortBy = null; else sortDir = -sortDir;
      } else { sortBy = c.key; sortDir = c.asc ? 1 : -1; }
    } else if (pick){
      const k = pick.dataset.col;
      if (k === 'ceil') state.showCeil = !state.showCeil; else OPTIONAL[k] = !OPTIONAL[k];
    } else return;
    if (lastFrontierRows) renderFrontierTable(lastFrontierRows, lastFrontierCurKey);
  });
}

/* ---- Chart H: the frontier as a picture — Terminal-Bench vs €/seat ------
   The table ranks by max users and prints a bill; the hardware line of
   the bill is a function of the GPU count alone, so users-vs-cost collapsed
   every row on one topology onto a band and said "run fewer GPUs". The buying question is
   what a seat costs against what the model can do: x = the bill with the
   row FULL divided by the users it then carries — its €/seat at capacity
   (log; seats span more than a decade). Not the bill at your load over your
   users: that is the GPU count again, and every row on a topology would
   price the same. y =
   the model's Terminal-Bench score on the version state.bench selects
   (research/terminal_bench.md; one value per model, so a model's rows stack
   on one horizontal line whose spread is the price of the topology choice). The two
   versions rank the top of this field differently — 2.1 is saturated, 4.0
   is not — so the toggle is part of the reading, not a preference. The Pareto-efficient set (no other
   row scores >= for <= money) is a staircase: for any capability floor,
   the cheapest seat. Rows that cannot carry the load have no seat price
   and are counted, not drawn. Drawn from EXACTLY the rows the table was
   handed (assembleFrontier's one commit point, and redrawCharts for theme
   flips), never from a half-rebuilt set. */
export let frontierChartGeom = null;
// short row name for a direct label: the model as its button reads (a full
// name is ~25 characters and three of them stack at a 560-wide viewBox) and
// the split as the DP×TP shorthand the split control uses
const FRONTIER_SHORT = { "27B": "Qwen3.8-27B", "35BA3B": "35B-A3B", "MM35": "Mistral-Med-3.5",
                         "GLM52": "GLM-5.3", "DSV4F": "DSv4-Flash", "DSV41F": "DSv4.1-Flash", "Q38FN": "Q3.8-Flash",
                         "GLM53F": "G5.3-Flash" };
// the row as the table and the tooltip print it: the model without its
// architecture tag and the split in the TP/DP shorthand. r.label keeps the
// Python-identical topology name (deploy card, harness, self-checks)
export function frontierRowName(r){
  // the arm tags modelFor appends ([NVFP4], [FP16 KV], [fp32 state], ...)
  // ride along: a row priced on a non-default arm must say so
  const tags = (r.label.split(' · ')[0].match(/\[[^\]]+\]/g) || []).join(' ');
  return CONFIG.MODELS[r.mk].name.replace(/\s*\(.*\)\s*$/, '') + (tags ? ' ' + tags : '') + ' · '
       + makeGrid(r.dp, r.tp, state.gpu).name.replace(' tensor-par', ' TP').replace(' data-par', ' DP');
       // (the un-suffixed name: the KV layout is a page-wide setting, not a row label)
}
function frontierShortLabel(r){
  const m = FRONTIER_SHORT[r.mk] || CONFIG.MODELS[r.mk].name.replace(/\s*\(.*\)\s*$/, '');
  const t = r.dp*r.tp === 1 ? '1 GPU' : r.dp === 1 ? `TP${r.tp}`
          : r.tp === 1 ? `DP${r.dp}` : `DP${r.dp}×TP${r.tp}`;
  return `${m} · ${t}`;
}
// NaN, never null, for a model without a run: isFinite(null) is true in JS
// (null coerces to 0), which would plot an unscored model at 0% instead of
// leaving it off the chart and printing '—' in the table. A MEASURED zero
// (two models score 0/198 on 4.0) is a score and must survive this — hence
// Number.isFinite and not a truthiness test.
export const frontierScore = (r, bench = state.bench) => {
  const q = (CONFIG.QUALITY[r.mk] || {})[bench];
  return Number.isFinite(q) ? q : NaN;
};
export function renderFrontierChart(rows, curKey){
  const box = document.getElementById('chartH'); if (!box) return;
  const tt = document.getElementById('ttH'); if (tt) tt.style.opacity = 0;
  const users = Math.max(1, state.users);
  // a seat price needs a ceiling to fill; a configuration that cannot carry
  // the load is not a choice and is counted in-chart rather than drawn
  const viable = rows.filter(r => isFinite(r.op.limit) && r.op.limit >= 1 && isFinite(r.eurSeat) && r.eurSeat > 0);
  const carries = viable.filter(r => r.op.fits);
  const live = carries.filter(r => isFinite(frontierScore(r)));
  const over = viable.length - carries.length, unscored = carries.length - live.length;
  const notViable = rows.length - viable.length;
  const perUser = r => r.eurSeat;
  if (!live.length){
    const W=560,H=120;
    const why = carries.length ? `no configuration that carries ${fmt(users,0)} users has a ${CONFIG.BENCHES[state.bench].name} score`
                               : `no configuration on this GPU can carry ${fmt(users,0)} users at these settings`;
    box.innerHTML = svgEl(`<text x="${W/2}" y="${H/2+4}" text-anchor="middle" class="axlbl" font-size="13">${esc(why)}</text>`,
                          W, H, 'Nothing to plot');
    frontierChartGeom = null; return;
  }
  // same width rule as chart G: act 3 panels are page-wide, so the viewBox
  // must track the paint width or the type scales with it
  const wide = (typeof window !== 'undefined' ? window.innerWidth : 1400) >= 900;
  const W = wide ? 1120 : 560, H = wide ? 420 : 360;
  // the top margin holds the score source and the census: one line side by
  // side when wide, two stacked lines when narrow
  const mL = wide ? 56 : 48, mR = wide ? 22 : 16, mT = wide ? 26 : 40, mB = wide ? 46 : 42;
  const pw=W-mL-mR, ph=H-mT-mB;
  const grid=cssv('--grid'), axis=cssv('--axis'), muted=cssv('--muted');
  const surface=cssv('--surface'), text=cssv('--text');
  const C = PLANNER_COLORS();
  const eurTick = t => t >= 1000 ? `€${fmt(t/1000, t % 1000 ? 1 : 0)}k` : t >= 10 ? `€${fmt(t,0)}` : `€${fmt(t,1)}`;
  // the Pareto-efficient set: nothing else scores as high for as little
  // per seat. Two rows on the same GPU count differ in price only by the
  // electricity term, and the power model carries ±20-25% (research/
  // power.md), so a bill within 1% is a TIE — otherwise three 1-GPU rows
  // at €315.0 / €315.4 / €315.9 all count as efficient and the higher
  // score among them does not win. Scores tie exactly (same model).
  const near = (a,b) => perUser(b) <= perUser(a)*1.01;
  const par = new Set(live.filter(a => !live.some(b => b !== a
      && frontierScore(b) >= frontierScore(a) && near(a,b)
      && (frontierScore(b) > frontierScore(a) || perUser(b) < perUser(a)))));
  const qs = live.map(r => frontierScore(r)*100), es = live.map(perUser);
  // x = € per seat (log), y = score (%): both grow away from the origin, so
  // the efficient set is the upper-left edge. Headroom above the top score
  // is for the direct labels, which sit to the upper left; the axis is a
  // percentage, so it never runs past 100. The low end floors at -5 rather
  // than 0: two models score a MEASURED 0.0% on 4.0, and clamping to 0 drew
  // their dots on top of the price ticks. No tick is labelled below 0 (the
  // loop starts at ceil(yLo/10)*10), so the slack is drawing room only.
  const yLo = Math.max(-5, Math.floor((Math.min(...qs)-4)/5)*5), yHi = Math.min(100, Math.max(...qs) + (wide ? 8 : 12));
  const xLo = Math.min(...es)*0.7, xHi = Math.max(...es)*1.5;
  const sx = logScale(xLo, xHi, mL, mL+pw), sy = linScale(yLo, yHi, mT+ph, mT);
  let g='';
  for (const t of logTicks(xLo,xHi)){
    const X=sx(t);
    g+=`<line x1="${X}" y1="${mT}" x2="${X}" y2="${mT+ph}" stroke="${grid}" stroke-width="1"/>`;
    g+=`<text class="axtick" x="${X}" y="${mT+ph+16}" text-anchor="middle">${eurTick(t)}</text>`;
  }
  for (let t=Math.ceil(yLo/10)*10; t<=yHi; t+=10){
    const Y=sy(t);
    g+=`<line x1="${mL}" y1="${Y}" x2="${mL+pw}" y2="${Y}" stroke="${grid}" stroke-width="1"/>`;
    g+=`<text class="axtick" x="${mL-8}" y="${Y+3}" text-anchor="end">${t}%</text>`;
  }
  // the staircase is the function "best score a seat budget of p buys":
  // flat at an efficient row's score from its price to the next efficient
  // row's price, then a jump up to that row's score. Lead-out: flat at the
  // top row's score to the right edge — a bigger budget buys nothing
  // better. No lead-in: below the cheapest price nothing carries the load.
  const stair = [...par].sort((a,b)=>frontierScore(a)-frontierScore(b));
  const P = r => [sx(perUser(r)), sy(frontierScore(r)*100)];
  if (stair.length){
    const [x0,y0] = P(stair[0]);
    let d = `M ${x0} ${y0}`;
    for (let i=1;i<stair.length;i++){
      const [x,y] = P(stair[i]);
      d += ` L ${x} ${P(stair[i-1])[1]} L ${x} ${y}`;
    }
    d += ` L ${mL+pw} ${P(stair[stair.length-1])[1]}`;
    g+=`<path d="${d}" fill="none" stroke="${muted}" stroke-width="1.5" stroke-linejoin="round" opacity="0.7"/>`;
  }
  // dots: dominated first (dimmed), then the efficient set, then the
  // selected configuration's ring on top of everything
  const pts = [];
  // a censored decode limit is a LOWER bound (the search stopped before the
  // floor): hollow, so the reader sees a bound, not a point. The seat price
  // is priced AT that limit, so it is an UPPER bound (both bill/users terms
  // fall as the true ceiling rises); the table and tooltip carry the '≤'.
  const dot = (r, on) => {
    const [x,y] = P(r), col = C[r.op.binding] || muted;
    pts.push({ x, y, color: col, r });
    const hollow = r.op.binding==='decode' && r.censored;
    return `<circle cx="${x}" cy="${y}" r="5" fill="${hollow?surface:col}" stroke="${hollow?col:surface}" stroke-width="2"${on?'':' opacity="0.35"'}/>`;
  };
  for (const r of live) if (!par.has(r)) g += dot(r, false);
  for (const r of stair) g += dot(r, true);
  const cur = live.find(r => r.key === curKey);
  if (cur){
    const [x,y] = P(cur);
    g+=`<circle cx="${x}" cy="${y}" r="10" fill="none" stroke="${text}" stroke-width="1.5" opacity="0.8"/>`;
    // an efficient selection is named by its own label below; a dominated
    // one would otherwise be an anonymous ring
    if (!par.has(cur))
      g+=`<text class="dlabel" x="${x+14}" y="${y+4}" text-anchor="start" fill="${muted}">you</text>`;
  }
  // direct labels on the efficient set only, to the UPPER LEFT of each dot:
  // a higher score for less money is empty of efficient dots by definition —
  // but not of dominated ones, and the 1% tolerance lets a dearer dot sit
  // just beside, so every drawn dot is an obstacle too. Labels that would
  // collide are pushed up in y order; one that would run past the left edge
  // flips to the lower right instead. Width is estimated (no layout pass in
  // an SVG string). A displaced label gets a hairline leader back to its
  // dot: two efficient rows at nearly the same score stack two labels, and
  // without the leader the reader cannot tell which name is which dot.
  const placed = pts.map(p => ({ x0: p.x-6, x1: p.x+6, y0: p.y-6, y1: p.y+6 }));
  const overlaps = (a,b) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
  for (const r of [...stair].sort((a,b)=>P(b)[1]-P(a)[1])){
    const [x,y] = P(r), name = frontierShortLabel(r) + (r.key===curKey ? ' (you)' : '');
    const w = name.length*5.9, h = 12;
    const flip = x - 9 - w < mL;
    const bx = flip ? x+9 : x-9-w, by = flip ? y+4 : y-18;
    let bb = { x0: bx, x1: bx+w, y0: by, y1: by+h };
    for (let k=0;k<12;k++){
      const hit = placed.find(q => overlaps(bb,q));
      if (!hit) break;
      bb = { ...bb, y0: hit.y0-2-h, y1: hit.y0-2 };
    }
    if (bb.y0 < mT){                                        // off the top: go below,
      bb = { ...bb, y0: y+6, y1: y+6+h };                    // pushing DOWN past what is placed
      for (let k=0;k<12;k++){
        const hit = placed.find(q => overlaps(bb,q));
        if (!hit) break;
        bb = { ...bb, y0: hit.y1+2, y1: hit.y1+2+h };
      }
    }
    placed.push(bb);
    if (bb.y0 !== by){
      const lx = flip ? bb.x0-3 : bb.x1+3, ly = bb.y1-6;
      g+=`<line x1="${x}" y1="${y}" x2="${lx}" y2="${ly}" stroke="${muted}" stroke-width="1" opacity="0.7"/>`;
    }
    g+=`<text class="dlabel" x="${flip?bb.x0:bb.x1}" y="${bb.y1-2}" text-anchor="${flip?'start':'end'}">${esc(name)}</text>`;
  }
  // the census and the score source sit in the top margin, outside the
  // plot, where no dot or label can land
  const notes = [`at ${fmt(users,0)} users`];
  // the plan grid holds six DP x TP shapes; the split control offers every
  // divisor, so a DP3 or TP6 selection matches no row and gets no ring —
  // say so rather than leave the reader hunting for "you"
  const curOver = viable.some(r => r.key === curKey && !r.op.fits);
  const curNotViable = rows.some(r => r.key === curKey && !viable.includes(r));
  if (over) notes.push(`${over} row${over>1?'s':''} cannot carry it${curOver?' (yours among them)':''}`);
  if (unscored) notes.push(`${unscored} unscored`);
  if (notViable) notes.push(`${notViable} not viable${curNotViable?' (yours among them)':''}`);
  if (!rows.some(r => r.key === curKey)) notes.push('your split is not in the grid');
  // a row scored from its vendor card (QUALITY[mk].source) is not an AA
  // measurement: the source line names it rather than label it AA
  const vendor = [...new Set(live.filter(r => CONFIG.QUALITY[r.mk].source).map(r => FRONTIER_SHORT[r.mk] || r.mk))];
  const bench = CONFIG.BENCHES[state.bench];
  const axisSrc = `scores: Artificial Analysis, ${bench.harness}` + (vendor.length ? `; ${vendor.join(', ')}: vendor card` : '');
  g+=`<text class="axtick" x="${mL}" y="${wide ? mT-10 : 12}" text-anchor="start">${esc(axisSrc)}</text>`;
  g+=`<text class="axtick" x="${wide ? mL+pw : mL}" y="${wide ? mT-10 : 26}" text-anchor="${wide ? 'end' : 'start'}">${esc(notes.join(' · '))}</text>`;
  g+=`<line x1="${mL}" y1="${mT+ph}" x2="${mL+pw}" y2="${mT+ph}" stroke="${axis}" stroke-width="1"/>`;
  g+=`<text class="axlbl" x="${mL+pw/2}" y="${H-6}" text-anchor="middle">€ per seat per month, configuration full (log)</text>`;
  g+=`<text class="axlbl" x="${12}" y="${mT+ph/2}" text-anchor="middle" transform="rotate(-90 12 ${mT+ph/2})">${esc(bench.name)}, pass@1</text>`;
  box.innerHTML = svgEl(g, W, H,
    `Every configuration that carries the load as monthly cost per seat at capacity versus ${bench.name} score, with the Pareto-efficient set joined as a staircase`);
  frontierChartGeom = { W,H,mL,mR,mT,pw,ph, pts, par, curKey };
}
