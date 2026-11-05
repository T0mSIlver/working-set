#!/usr/bin/env python3
"""Refresh data/external/inferencex.json from SemiAnalysis InferenceX.
Usage: python3 scripts/external_inferencex.py fetch   # pull API JSON into $IX_RAW (public API, no auth)
       python3 scripts/external_inferencex.py build   # rewrite data/external/inferencex.json
Method and caveats: research/sources/inferencex.md.
Source of truth: https://inferencex.semianalysis.com/api/v1/benchmarks?model=<display model>
(openapi: /api/openapi.json). Rows keep their per-curve `date`; no date param = latest."""
import datetime, json, sys, os, glob, urllib.request, collections
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get('IX_RAW', '/tmp/inferencex-raw'); os.makedirs(RAW, exist_ok=True)
BASE = 'https://inferencex.semianalysis.com'
MODELS = ['DeepSeek-V4.1-Flash','GLM-5.2','GLM-5.3-Flash','Qwen3.8-Flash-Next','Qwen3.8-27B','Qwen3.8-27B-Eager',
          'Qwen-3.5-397B-A17B','DeepSeek-V4-Pro','DeepSeek-R1-0528','gpt-oss-120b','Kimi-K3',
          'Llama-3.3-70B-Instruct-FP8','MiniMax-M3.1-Flash','Muse-Spark-Open','GLM-5']
OUT = os.path.join(HERE, '..', 'data', 'external', 'inferencex.json')
READ = datetime.date.today().isoformat()
REPO_COMMIT = '6481779aae17f6f5e7c1a6057ac41d93ed8fcff8'      # InferenceX, 2026-11-04
APP_COMMIT = '42da605b6d09de6d7939b2f728e74b8b9ea5e57f'      # InferenceX-app, 2026-11-04

def get(path, out):
    with urllib.request.urlopen(BASE + path, timeout=180) as r: data = r.read()
    open(os.path.join(RAW, out), 'wb').write(data)

def fetch():
    get('/api/openapi.json', '_openapi.json'); get('/api/v1/views/options', '_options.json')
    get('/api/v1/views/model-architectures', '_arch.json')
    for m in MODELS: get('/api/v1/benchmarks?model=' + m, m + '.json')

GPUS = ['h100','h200','b200','b300','gb200','gb300','mi355x']
GPU_NAME = {'h100':'H100','h200':'H200','b200':'B200','b300':'B300','gb200':'GB200','gb300':'GB300','mi355x':'MI355X'}
COSTH = {'h100':1.17,'h200':1.22,'b200':1.73,'b300':2.26,'gb200':1.86,'gb300':2.31,'mi355x':1.5}  # HW_REGISTRY.costh
# display model, ws key, dashboard name
OVERLAP = {'dsv41flash':('DeepSeek-V4.1-Flash','DSV41F'), 'glm5.3':('GLM-5.3','GLM52'), 'qwen3.8next':('Qwen3.8-Flash-Next','Q38FN')}
OTHER = {'dsr1':'DeepSeek-R1-0528','qwen3.5':'Qwen3.5-397B-A17B','llama70b':'Llama-3.3-70B-Instruct-FP8',
         'gptoss120b':'gpt-oss-120b','dsv4':'DeepSeek-V4-Pro'}
# non-overlap models: which workload to extract
OTHER_WL = {'dsr1':['8k/1k'],'qwen3.5':['8k/1k'],'llama70b':['8k/1k'],'gptoss120b':['8k/1k'],'dsv4':['agentic']}

def load():
    rows = []
    for f in glob.glob(os.path.join(RAW, '*.json')):
        if os.path.basename(f).startswith('_'): continue
        rows += json.load(open(f))
    return rows

def wl(r):
    return 'agentic' if r['benchmark_type'] == 'agentic_traces' else '%dk/%dk' % (r['isl']//1024, r['osl']//1024)

def ngpu(r):
    m = r['metrics']
    if m.get('total_tput_tps') and m.get('tput_per_gpu'): return round(m['total_tput_tps']/m['tput_per_gpu'])
    if r['disagg']: return r['num_prefill_gpu'] + r['num_decode_gpu']
    return r['num_decode_gpu'] if r['is_multinode'] else r['decode_tp']

def par(r):
    p = 'TP%d' % r['decode_tp']
    if r['decode_ep'] and r['decode_ep'] > 1: p += ' EP%d' % r['decode_ep']
    if r['decode_dp_attention']: p += ' DPA'
    if r['disagg']: p = 'disagg prefill %s x%s / decode %s x%s' % (r['prefill_tp'], r['prefill_num_workers'], p, r['decode_num_workers'])
    return p

def curve(r):
    """One curve = one serving config swept over concurrency."""
    return '%s-%s-%s-%s-%s-%dg' % (r['model'], r['hardware'], r['framework'], r['precision'],
                                   par(r).replace(' ', ''), ngpu(r))

def record(r, tag, group, cid=None):
    m = r['metrics']; ag = r['benchmark_type'] == 'agentic_traces'; n = ngpu(r)
    ov = r['model'] in OVERLAP
    name, ws = OVERLAP[r['model']] if ov else (OTHER.get(r['model'], r['model']), None)
    mets = [{'name':'concurrency','value':r['conc'],'unit':'sessions' if ag else 'requests'},
            {'name':'tok_s_per_user','value':round(m['median_intvty'],1),'unit':'tok/s','at':'median interactivity = 1/median TPOT (agentic: 1/median ITL)'},
            {'name':'tpot_ms','value':round(m['median_tpot']*1000,2),'unit':'ms','at':'median'},
            {'name':'ttft_p50_s','value':round(m['median_ttft'],3),'unit':'s'},
            {'name':'tok_s_per_gpu','value':round(m['tput_per_gpu'],1),'unit':'tok/s','at':'input+output tokens per GPU, as the dashboard plots it'}]
    if ag:
        mets.append({'name':'out_tok_s_per_gpu','value':round(m['output_tput_per_gpu'],1),'unit':'tok/s'})
        mets.append({'name':'cache_hit_rate','value':round(m.get('server_gpu_cache_hit_rate',0),4),'unit':'fraction','at':'server_gpu_cache_hit_rate'})
    else:
        mets.append({'name':'usd_per_mtok','value':round(COSTH[r['hardware']]/(m['tput_per_gpu']*3600/1e6),3),'unit':'USD/Mtok','kind':'derived',
                     'at':'costh/(tput_per_gpu*3600/1e6), all tokens, costh=%.2f USD/GPU-h' % COSTH[r['hardware']]})
    mets.append({'name':'usd_per_gpu_h','value':COSTH[r['hardware']],'unit':'USD/GPU-h','at':'HW_REGISTRY.costh (owning, hyperscaler volume)'})
    return {
      'id': 'inferencex/%s-%s' % (cid, tag) if cid else 'inferencex/%s-%s-%s-%s-%s' % (r['model'], r['hardware'], r['framework'], wl(r).replace('/','x'), tag),
      'url': '%s/api/v1/benchmarks?model=%s' % (BASE, name if not ov or r['model']!='glm5.3' else 'GLM-5.2'),
      'read': READ, 'published': r['date'], 'kind': 'measured',
      # a vendor-tuned curve point with its full config: fits the decode step (calibrate)
      'use': 'calibrate',
      'config': {'model': name, 'ws_model': ws, 'gpu': GPU_NAME[r['hardware']], 'n_gpus': n, 'weights': r['precision'], 'kv': None,
                 'engine': '%s %s' % (r['framework'], (r['image'] or 'n/a').split('@')[0].split(':')[-1]), 'parallel': par(r),
                 'spec_decode': r['spec_method'], 'max_ctx': None,
                 'extra': {'curve': cid, 'disagg': r['disagg'], 'multinode': r['is_multinode'], 'row_id': r['id'], 'run_url': r['run_url'],
                           'offload': r.get('offload_mode'), 'kv_offload': m.get('kv_offloading')}},
      'workload': {'name': 'AgentX trace replay' if ag else 'InferenceX single-turn random', 'isl': None if ag else r['isl'],
                   'osl': None if ag else r['osl'], 'prefix_cache': bool(ag), 'extra': {}},
      'metrics': mets,
      'quote': 'API rows[id=%s] (model=%s hardware=%s framework=%s conc=%s): metrics.median_intvty=%s, metrics.median_tpot=%s, metrics.tput_per_gpu=%s' %
               (r['id'], r['model'], r['hardware'], r['framework'], r['conc'], m['median_intvty'], m['median_tpot'], m['tput_per_gpu']),
      'note': (('Point at concurrency %d of curve %s. ' % (r['conc'], cid)) if cid else
               '%s point of group %s. ' % ('Lowest-latency point with conc<=8' if tag=='low' else 'Max-throughput point (%s)' % ('out_tok_s_per_gpu' if ag else 'tok_s_per_gpu'), group)
               + 'n_gpus = total_tput_tps/tput_per_gpu (agentic) or topology fields (single-turn). Row date %s is the curve date; API has no pinned URL, rows keyed by extra.row_id.' % r['date'])}

def build():
    rows = [r for r in load() if r['hardware'] in GPUS]
    groups = collections.defaultdict(list)
    for r in rows:
        w = wl(r)
        if r['model'] in OVERLAP: key = (r['model'], r['hardware'], r['framework'], w)
        elif r['model'] in OTHER_WL and w in OTHER_WL[r['model']]: key = (r['model'], r['hardware'], 'any', w)
        else: continue
        groups[key].append(r)
    recs = []
    for key, g in sorted(groups.items()):
        ag = key[3] == 'agentic'
        if key[0] in OVERLAP and ag:
            # our models: every point of every single-node curve, so a page can
            # draw per-user speed against sessions next to the model's curve
            for r in sorted(g, key=lambda r: (curve(r), r['conc'])):
                if r['disagg'] or r['is_multinode']: continue
                recs.append(record(r, 'c%d' % r['conc'], '/'.join(key), cid=curve(r)))
            continue
        lowc = [r for r in g if r['conc'] <= 8]
        score = (lambda r: r['metrics']['output_tput_per_gpu']) if ag else (lambda r: r['metrics']['tput_per_gpu'])
        if lowc: recs.append(record(max(lowc, key=lambda r: r['metrics']['median_intvty']), 'low', '/'.join(key)))
        mx = max(g, key=score)
        if not (lowc and mx['id'] == max(lowc, key=lambda r: r['metrics']['median_intvty'])['id']): recs.append(record(mx, 'max', '/'.join(key)))
    # MBU records
    recs += mbu_records(load())
    recs += golden_al_records()
    src = {'id':'inferencex','name':'SemiAnalysis InferenceX v2 (formerly InferenceMAX)','publisher':'SemiAnalysis',
           'url':'https://inferencex.semianalysis.com/','license':'Apache License 2.0 for the InferenceX repo (LICENSE @ %s); the InferenceX-app repo (dashboard + API) is GPL-3.0 (LICENSE @ %s)' % (REPO_COMMIT[:7], APP_COMMIT[:7]),
           'method':'Nightly sweeps of open models on vendor GPUs with SGLang/vLLM/TRT-LLM (and AMD ATOM/MoRI); single-turn random 8k/1k and AgentX trace replay; tok/s/GPU vs tok/s/user across concurrency; cost = costh USD/GPU-h / (tok/s/GPU*3600/1e6).',
           'kind':'measured','refresh_days':14,'redistribute':'cite',
           'terms':'InferenceX repo Apache-2.0; results come from its public API (app repo GPL-3.0) with no separate data licence, so cited with attribution to SemiAnalysis.'}
    for r in recs:
        if len(r['quote']) > 600: r['quote'] = r['quote'][:597] + '...'
    with open(OUT, 'w') as f:
        json.dump({'source':src,'records':recs}, f, indent=1, ensure_ascii=False); f.write('\n')
    print(len(recs), 'records ->', OUT)

# weights-only decode MBU for conc-1/4 non-spec single-node single-turn 8k/1k points
BW = {'h100':3.35,'h200':4.8,'b200':8.0,'b300':8.0}
# (row filter) -> (display, total_B, active_B, bytes_per_param, bpp_note)
MBU_PICK = [
  ('qwen3.5','h100','sglang','fp8',8,1, 397,17,1.0), ('qwen3.5','b200','sglang','fp8',8,1, 397,17,1.0),
  ('dsr1','b200','sglang','fp8',8,1, 671,37,1.0), ('dsr1','b300','sglang','fp8',8,1, 671,37,1.0),
  ('dsr1','b200','sglang','fp4',4,1, 671,37,0.5625),
  ('llama70b','h200','vllm','fp8',1,4, 70,70,1.0), ('llama70b','h100','vllm','fp8',2,4, 70,70,1.0), ('llama70b','b200','trt','fp8',1,4, 70,70,1.0),
]
def mbu_rows(allrows):
    out = []
    for mdl,hw,fw,prec,tp,conc,tot,act,bpp in MBU_PICK:
        c = [r for r in allrows if r['model']==mdl and r['hardware']==hw and r['framework']==fw and r['precision']==prec
             and r['benchmark_type']=='single_turn' and r['isl']==8192 and r['spec_method']=='none' and not r['disagg'] and not r['is_multinode']
             and r['decode_tp']==tp and r['decode_ep']==1 and r['conc']==conc]
        if not c: print('MBU row missing', mdl, hw, fw, prec, tp, conc); continue
        r = max(c, key=lambda r: r['date']); tpot = r['metrics']['median_tpot']
        gb = act*bpp; mbu = gb*1e9/(tpot*tp*BW[hw]*1e12)
        out.append((r, tot, act, bpp, gb, tpot, mbu, tp))
    return out
def mbu_records(allrows):
    recs = []
    for r, tot, act, bpp, gb, tpot, mbu, tp in mbu_rows(allrows):
        recs.append({'id':'inferencex/derived/mbu-%s-%s-tp%d-%s-conc%d' % (r['model'], r['hardware'], tp, r['precision'], r['conc']),
          'url':'%s/api/v1/benchmarks?model=%s' % (BASE, {'dsr1':'DeepSeek-R1-0528','qwen3.5':'Qwen-3.5-397B-A17B','llama70b':'Llama-3.3-70B-Instruct-FP8'}[r['model']]),
          'read':READ,'published':r['date'],'kind':'derived','use':'check',
          'config':{'model':OTHER[r['model']],'ws_model':None,'gpu':GPU_NAME[r['hardware']],'n_gpus':tp,'weights':r['precision'],'kv':None,
                    'engine':'%s %s' % (r['framework'], (r['image'] or 'n/a').split('@')[0].split(':')[-1]),'parallel':'TP%d'%tp,'spec_decode':'none','max_ctx':None,
                    'extra':{'row_id':r['id'],'total_params_b':tot,'active_params_b':act,'bytes_per_param':bpp,'hbm_tb_s':BW[r['hardware']]}},
          'workload':{'name':'InferenceX single-turn random','isl':8192,'osl':1024,'prefix_cache':False,'extra':{}},
          'metrics':[{'name':'concurrency','value':r['conc'],'unit':'requests'},{'name':'tpot_ms','value':round(tpot*1000,2),'unit':'ms','kind':'measured'},
                     {'name':'mbu','value':round(mbu,3),'unit':'fraction','kind':'derived','at':'weights-only lower bound'}],
          'quote':'API rows[id=%s].metrics.median_tpot=%s; params from /api/v1/views/model-architectures (totalParams=%s, activeParams=%s)' % (r['id'], r['metrics']['median_tpot'], tot, act),
          'note':'derived: MBU = active_params*bytes_per_param / (median_tpot * n_gpus * HBM_BW). Weights only (no KV, no activations, no scales), so a lower bound; conc=%d with MoE reads only the routed experts at conc 1. bytes/param is our assumption (fp8=1, nvfp4~0.5625).' % r['conc']})
    return recs

# Draft acceptance InferenceX pins in its agentic runs ("golden AL", measured
# with SPEED-Bench coding). Needs a clone: IX_REPO=/path/to/InferenceX.
GOLDEN = {'glm-5.3-nvfp4__mtp__native__default-default.yaml': ('GLM-5.3', 'GLM52', 'mtp'),
          'deepseek-v4.1-flash__dspark__native__probabilistic-block.yaml': ('DeepSeek-V4.1-Flash', 'DSV41F', 'dspark'),
          'qwen3.8-flash-next-fp8__mtp__native__default-default.yaml': ('Qwen3.8-Flash-Next', 'Q38FN', 'mtp')}
def golden_al_records():
    repo = os.environ.get('IX_REPO')
    if not repo:
        print('IX_REPO unset: no acceptance records'); return []
    d = os.path.join(repo, 'inferencex-e2e', 'infx', 'golden_al_distribution')
    recs = []
    for fn, (name, ws, method) in GOLDEN.items():
        lines = open(os.path.join(d, fn)).read().splitlines()
        run = next((l.split(': ', 1)[1] for l in lines if l.startswith('# Source GitHub Actions run:')), '')
        mode, mets = None, []
        for l in lines:
            if l and not l.startswith(' ') and not l.startswith('#') and not l.startswith(fn.split('__')[0]):
                mode = None                      # a later top-level block (client settings)
            if l in ('  thinking_on:', '  thinking_off:'): mode = l.strip()[:-1]
            elif mode and l.startswith('    ') and l.strip()[:1].isdigit():
                k, v = l.strip().split(':', 1)
                if v.strip() not in ('', 'N/A'):
                    mets.append({'name': 'acceptance_length', 'value': float(v), 'unit': 'tokens per step',
                                 'at': '%s draft tokens, %s' % (k, mode.replace('_', ' '))})
        recs.append({'id': 'inferencex/golden-al-%s' % fn.split('__')[0], 'kind': 'measured', 'use': 'calibrate',
          'url': 'https://github.com/SemiAnalysisAI/InferenceX/blob/%s/inferencex-e2e/infx/golden_al_distribution/%s' % (REPO_COMMIT, fn),
          'read': READ, 'config': {'model': name, 'ws_model': ws, 'gpu': None, 'spec_decode': method, 'extra': {'run_url': run}},
          'workload': {'name': 'SPEED-Bench coding', 'osl': 4096, 'extra': {}}, 'metrics': mets,
          'quote': '%s: %s' % (fn, '; '.join('%s=%s' % (m['at'], m['value']) for m in mets[:4])),
          'note': 'Acceptance length per draft depth, including the verified token. InferenceX pins this value in its agentic throughput runs, so per-user speeds there assume it.'})
    return recs

if __name__ == '__main__':
    {'fetch':fetch,'build':build}[sys.argv[1]]()
