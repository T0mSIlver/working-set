/* ============================================================================
   SVG chart helpers
   ========================================================================== */
export const NS = "http://www.w3.org/2000/svg";
export function cssv(name){ return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
export function fmt(x, d){ return x.toLocaleString("en-US",{maximumFractionDigits:d??0, minimumFractionDigits:d??0}); }
export function esc(s){ return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/"/g,"&quot;"); }

// generic scales
export function linScale(d0,d1,r0,r1){ const m=(r1-r0)/(d1-d0||1); return x=>r0+(x-d0)*m; }
// log10 scale; affine in ln(x), so a straight screen segment between two
// samples is exactly the geometric interpolation of their values (see interpAt)
export function logScale(d0,d1,r0,r1){
  const l0=Math.log(d0), m=(r1-r0)/((Math.log(d1)-l0)||1);
  return x=>r0+(Math.log(Math.max(x,d0*1e-9))-l0)*m;
}
// 1-2-5 "nice" ticks for a LINEAR axis from 0 to hi. max/N divisions give
// ticks like 0/47/93/140/187, off which no value can be estimated; this gives
// 0/50/100/150/200.
function niceStep(raw){
  // classic nice-number rounding: pick 1/2/5/10 x 10^n by the MANTISSA, not
  // the first ladder entry above `raw` — the latter systematically overshoots
  // and leaves narrow panels with one or two ticks
  if (!(raw > 0)) return 1;
  const exp = Math.floor(Math.log10(raw)), f = raw/Math.pow(10, exp);
  const m = f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10;
  return m * Math.pow(10, exp);
}
export function niceTicks(hi, target){
  if (!(hi > 0)) return [0];
  const step = niceStep(hi/(target||5));
  const out = [];
  for (let v=0; v <= hi + step*1e-9; v += step) out.push(v);
  return out;
}
// 1-2-5 ticks inside an arbitrary [lo,hi] that need not start at zero — for
// panels framed on their data, where 4-way division produces labels like
// 42% / 43% / 45% / 46% that look mis-spaced once rounded
export function niceTicksIn(lo, hi, target){
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const step = niceStep(span/(target||4));
  const out = [];
  for (let v = Math.ceil(lo/step - 1e-9)*step; v <= hi + step*1e-9; v += step) out.push(v);
  return out.length ? out : [lo, hi];
}
// 1-2-5 decade ticks inside [lo,hi]; falls back to a finer set for narrow ranges
export function logTicks(lo,hi){
  const build=ms=>{
    const out=[];
    for(let e=Math.floor(Math.log10(lo)); e<=Math.ceil(Math.log10(hi)); e++)
      for(const m of ms){ const v=m*Math.pow(10,e); if(v>=lo&&v<=hi) out.push(v); }
    return out;
  };
  const coarse=build([1,2,5]);
  return coarse.length>=4 ? coarse : build([1,1.5,2,3,5,7]);
}

// build an svg element from an html string; label = the accessible name a
// screen reader announces for the role="img" chart
export function svgEl(inner, vbW, vbH, label){
  return `<svg class="chart" viewBox="0 0 ${vbW} ${vbH}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="${esc(label||'chart')}">${inner}</svg>`;
}
