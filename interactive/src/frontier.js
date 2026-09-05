import { CONFIG, makeGrid } from './config.js';
import { state } from './state.js';
import { cssv, esc, fmt, linScale, logScale, logTicks, svgEl } from './svg.js';
import { PLANNER_COLORS, PLANNER_LABEL } from './planner.js';

/* ---- The frontier table ---- */
/* A DECISION table: the row's ranking columns are the verdict at YOUR load,
   the max-user count, what binds, and the headroom multiple — the four raw
   ceiling columns that used to force the comparison onto the reader now sit
   behind a toggle (state.showCeil). Cached rows re-render on toggle. */
export let lastFrontierRows = null, lastFrontierCurKey = null;
export function renderFrontierTable(rows, curKey){
  lastFrontierRows = rows; lastFrontierCurKey = curKey;
  const C = PLANNER_COLORS();
  // status colors as var() references, so they track a theme flip without a
  // re-render (the bind column's chart palette still needs the redraw hook)
  const good='var(--good)', warn='var(--warn)', crit='var(--crit)', muted='var(--muted)';
  const ceilHead = state.showCeil
    ? `<th class="num">cache</th><th class="num">decode</th>`
      + `<th class="num">latency</th><th class="num">saturation</th>` : '';
  const head = `<tr><th>configuration</th><th class="num">TB 2.1</th><th>your load</th><th class="num">max users</th>`
             + `<th>binds on</th><th class="num">headroom</th>${ceilHead}`
             + `<th class="num">B*</th><th class="num">€/mo</th><th class="num">€/seat</th></tr>`;   // .num headers right-align over their digits
  const body = rows.map(r=>{
    const you = r.key===curKey ? ' class="you"' : '';
    const viable = r.op.limit >= 1;
    const fits = viable && r.op.fits;
    const ceilCells = state.showCeil ? ['cache','decode','latency','saturation']
      .map(k=>`<td class="num"${k===r.op.binding?` style="color:${C[k]};font-weight:650"`:''}>`
             + `${k==='decode'&&r.censored?'≥ ':''}`
             + `${isFinite(r.op.ceilings[k])?fmt(r.op.ceilings[k],0):'—'}</td>`).join('') : '';
    const why = r.op.binding === 'latency'
      ? 'cannot meet the TTFT budget at any load'
      : (r.op.binding === 'saturation' ? 'prefill saturates before one user'
      : `${PLANNER_LABEL[r.op.binding]} allows under one user`);
    const room = viable ? r.op.limit/r.op.users : 0;
    // a censored decode search is a floor, not an estimate — carry the '≥'
    // through every figure derived from it, not just the headline
    const cen = r.op.binding==='decode' && r.censored ? '≥ ' : '';
    const q = frontierScore(r);
    return `<tr${you}><td>${esc(frontierRowName(r))}</td>`
         // the model's score, not the row's: every split of a model shares it
         + `<td class="num">${isFinite(q) ? fmt(q*100,1)+'%' : '—'}</td>`
         + (viable
            ? `<td class="v" style="color:${fits?good:crit}">${fits?'✓ fits':'✗ over'}</td>`
            : `<td class="v" style="color:${muted}">—</td>`)
         + (viable
            ? `<td class="num">${cen}${fmt(r.op.limit,0)}${r.reps>1?` <span style="color:${muted}">(${cen}${fmt(r.op.limit/r.reps,0)}/grp)</span>`:''}</td>`
            : `<td class="num" style="color:${muted}">not viable<br><span style="font-size:10.5px">${esc(why)}</span></td>`)
         + `<td class="bind" style="color:${C[r.op.binding]}">${esc(PLANNER_LABEL[r.op.binding])}</td>`
         + (viable
            ? `<td class="num" style="color:${fits?(room>1.25?good:warn):crit}">×${fmt(room, room<10?1:0)}${cen?'+':''}</td>`
            : `<td class="num" style="color:${muted}">—</td>`)
         + ceilCells
         + `<td class="num">${fmt(r.bstar,1)}</td>`
         // energy is meaningful only where the load can actually be served
         + `<td class="num"${viable&&fits?'':` style="color:${muted}"`}>${viable ? fmt(r.eur,0) : '—'}</td>`
         // chart H's y: the bill with the row full, per user it then carries
         + `<td class="num"${viable&&fits?'':` style="color:${muted}"`}>${viable && isFinite(r.eurSeat) ? fmt(r.eurSeat,0) : '—'}</td></tr>`;
  }).join('');
  document.getElementById('frontierTable').innerHTML =
    `<div class="ftable-wrap"><table class="ftable">${head}${body}</table></div>`;
}

/* ---- Chart H: the frontier as a picture — €/user vs Terminal-Bench ------
   The table ranks by max users and prints a bill; the hardware line of
   the bill is a function of the GPU count alone, so users-vs-cost collapsed
   every row on one topology onto a band and said "run fewer GPUs". The buying question is
   what a seat costs against what the model can do: y = the bill with the
   row FULL divided by the users it then carries — its €/seat at capacity
   (log; seats span more than a decade). Not the bill at your load over your
   users: that is the GPU count again, and every row on a topology would
   price the same. x =
   the model's Terminal-Bench 2.1 score (research/terminal_bench.md; one
   value per model, so a model's rows stack in a column and the column is
   the price of the topology choice). The Pareto-efficient set (no other
   row scores >= for <= money) is a staircase: for any capability floor,
   the cheapest seat. Rows that cannot carry the load have no seat price
   and are counted, not drawn. Drawn from EXACTLY the rows the table was
   handed (assembleFrontier's one commit point, and redrawCharts for theme
   flips), never from a half-rebuilt set. */
export let frontierChartGeom = null;
// short row name for a direct label: the model as its button reads (a full
// name is ~25 characters and three of them stack at a 560-wide viewBox) and
// the split as the DP×TP shorthand the split control uses
const FRONTIER_SHORT = { "27B": "Qwen3.6-27B", "35BA3B": "35B-A3B", "MM35": "Mistral-Med-3.5",
                         "GLM52": "GLM-5.2", "DSV4F": "DSv4-Flash", "Q38FN": "Q3.8-Flash",
                         "GLM53F": "G5.3-Flash" };
// the row as the table and the tooltip print it: the model without its
// architecture tag and the split in the TP/DP shorthand. r.label keeps the
// Python-identical topology name (deploy card, harness, self-checks)
export function frontierRowName(r){
  return CONFIG.MODELS[r.mk].name.replace(/\s*\(.*\)\s*$/, '') + ' · '
       + makeGrid(r.dp, r.tp, state.gpu).name.replace(' tensor-par', ' TP').replace(' data-par', ' DP');
}
function frontierShortLabel(r){
  const m = FRONTIER_SHORT[r.mk] || CONFIG.MODELS[r.mk].name.replace(/\s*\(.*\)\s*$/, '');
  const t = r.dp*r.tp === 1 ? '1 GPU' : r.dp === 1 ? `TP${r.tp}`
          : r.tp === 1 ? `DP${r.dp}` : `DP${r.dp}×TP${r.tp}`;
  return `${m} · ${t}`;
}
export const frontierScore = r => (CONFIG.QUALITY[r.mk] || {}).tb21;
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
    const why = carries.length ? `no configuration that carries ${fmt(users,0)} users has a Terminal-Bench score`
                               : `no configuration on this GPU can carry ${fmt(users,0)} users at these settings`;
    box.innerHTML = svgEl(`<text x="${W/2}" y="${H/2+4}" text-anchor="middle" class="axlbl" font-size="13">${esc(why)}</text>`,
                          W, H, 'Nothing to plot');
    frontierChartGeom = null; return;
  }
  // same width rule as chart G: act 3 panels are page-wide, so the viewBox
  // must track the paint width or the type scales with it
  const wide = (typeof window !== 'undefined' ? window.innerWidth : 1400) >= 900;
  const W = wide ? 1120 : 560, H = wide ? 400 : 330;
  const mL = wide ? 64 : 56, mR = wide ? 22 : 16, mT = 18, mB = wide ? 46 : 42;
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
  // right-hand slack is for the direct labels, which sit to the upper right;
  // the axis is a percentage, so it never runs past 100
  const xLo = Math.max(0, Math.floor((Math.min(...qs)-4)/5)*5), xHi = Math.min(100, Math.max(...qs) + (wide ? 8 : 14));
  const yLo = Math.min(...es)*0.7, yHi = Math.max(...es)*1.5;
  const sx = linScale(xLo, xHi, mL, mL+pw), sy = logScale(yLo, yHi, mT+ph, mT);
  let g='';
  for (const t of logTicks(yLo,yHi)){
    const Y=sy(t);
    g+=`<line x1="${mL}" y1="${Y}" x2="${mL+pw}" y2="${Y}" stroke="${grid}" stroke-width="1"/>`;
    g+=`<text class="axtick" x="${mL-8}" y="${Y+3}" text-anchor="end">${eurTick(t)}</text>`;
  }
  for (let t=Math.ceil(xLo/10)*10; t<=xHi; t+=10){
    const X=sx(t);
    g+=`<line x1="${X}" y1="${mT}" x2="${X}" y2="${mT+ph}" stroke="${grid}" stroke-width="1"/>`;
    g+=`<text class="axtick" x="${X}" y="${mT+ph+16}" text-anchor="middle">${t}%</text>`;
  }
  // the staircase is the function "cheapest seat that scores at least q":
  // flat at a row's price up to its score, then a jump to the next
  // efficient row's price. Each step is vertical at x_{i-1} (past that
  // score, the cheaper row no longer qualifies) then horizontal to x_i.
  // Lead-in: horizontal at the cheapest row's price from the left edge —
  // for any lower floor it is still the cheapest. No lead-out: nothing
  // scores higher than the last row, so the function ends there.
  const stair = [...par].sort((a,b)=>frontierScore(a)-frontierScore(b));
  const P = r => [sx(frontierScore(r)*100), sy(perUser(r))];
  if (stair.length){
    const [x0,y0] = P(stair[0]);
    let d = `M ${mL} ${y0} L ${x0} ${y0}`;
    for (let i=1;i<stair.length;i++){
      const [x,y] = P(stair[i]);
      d += ` L ${P(stair[i-1])[0]} ${y} L ${x} ${y}`;
    }
    g+=`<path d="${d}" fill="none" stroke="${muted}" stroke-width="1.5" stroke-linejoin="round" opacity="0.7"/>`;
  }
  // dots: dominated first (dimmed), then the efficient set, then the
  // selected configuration's ring on top of everything
  const pts = [];
  // a censored decode limit is a LOWER bound (the search stopped before the
  // floor): hollow, so the reader sees a bound, not a point. The seat price
  // does not depend on it (the bill is at your load), but "fits" does.
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
  // direct labels on the efficient set only, to the LOWER RIGHT of each dot:
  // a higher score for less money (the price axis grows upward) is empty of
  // efficient dots by definition — but not of dominated ones, and the 1%
  // tolerance lets a dearer dot sit just above, so every drawn dot is an
  // obstacle too. Labels that would collide are pushed down in y order; one
  // that would run past the right edge flips to the upper left instead.
  // Width is estimated (no layout pass in an SVG string). A displaced label
  // gets a hairline leader back to its dot: two efficient rows at nearly the
  // same price stack two labels, and without the leader the reader cannot
  // tell which name is which dot.
  const placed = pts.map(p => ({ x0: p.x-6, x1: p.x+6, y0: p.y-6, y1: p.y+6 }));
  const overlaps = (a,b) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
  for (const r of [...stair].sort((a,b)=>P(a)[1]-P(b)[1])){
    const [x,y] = P(r), name = frontierShortLabel(r) + (r.key===curKey ? ' (you)' : '');
    const w = name.length*5.9, h = 12;
    let end = x + 9 + w > W - mR;
    let bx = end ? x-9-w : x+9, by = end ? y-16 : y+4;
    let bb = { x0: bx, x1: bx+w, y0: by, y1: by+h };
    for (let k=0;k<12;k++){
      const hit = placed.find(q => overlaps(bb,q));
      if (!hit) break;
      bb = { ...bb, y0: hit.y1+2, y1: hit.y1+2+h };
    }
    if (bb.y1 > mT+ph) bb = { ...bb, y0: y-16, y1: y-4 };   // off the bottom: go above
    placed.push(bb);
    if (bb.y0 !== by){
      const lx = end ? bb.x1+3 : bb.x0-3, ly = bb.y1-6;
      g+=`<line x1="${x}" y1="${y}" x2="${lx}" y2="${ly}" stroke="${muted}" stroke-width="1" opacity="0.7"/>`;
    }
    g+=`<text class="dlabel" x="${end?bb.x1:bb.x0}" y="${bb.y1-2}" text-anchor="${end?'end':'start'}">${esc(name)}</text>`;
  }
  // the census sits inside the plot, top right: a higher score for less
  // money than every efficient row is empty, and the footer already holds
  // the axis title at the narrow width
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
  g+=`<text class="axtick" x="${mL+pw-4}" y="${mT+11}" text-anchor="end">${esc(notes.join(' · '))}</text>`;
  g+=`<line x1="${mL}" y1="${mT+ph}" x2="${mL+pw}" y2="${mT+ph}" stroke="${axis}" stroke-width="1"/>`;
  g+=`<text class="axlbl" x="${mL+pw/2}" y="${H-6}" text-anchor="middle">Terminal-Bench 2.1, pass@1 (Artificial Analysis)</text>`;
  g+=`<text class="axlbl" x="${12}" y="${mT+ph/2}" text-anchor="middle" transform="rotate(-90 12 ${mT+ph/2})">€ per seat per month, configuration full (log)</text>`;
  box.innerHTML = svgEl(g, W, H,
    'Every configuration that carries the load as Terminal-Bench score versus monthly cost per seat at capacity, with the Pareto-efficient set joined as a staircase');
  frontierChartGeom = { W,H,mL,mR,mT,pw,ph, pts, par, curKey };
}
