#!/usr/bin/env python
"""Measure per-epoch training time, peak GPU memory and trainable parameters for each variant."""
import os, sys, sys, time, json, subprocess, re
VARIANTS=[('XSimGCL (backbone)','XSimGCLg_w00'),('PT4Rec','PTbase_XSim_nofz'),
          ('Ours$_{-g}$','OURS_XSim_nofz'),('Ours (+DAGR)','OURSgeom_w2')]
PY = sys.executable
out={}
for name,tag in VARIANTS:
    cfg=f'conf/grid/{tag}__douban-book.yaml'
    if not os.path.exists(cfg):
        subprocess.run([PY,'scripts/run_grid.py','--models',tag,'--datasets','douban-book','--seeds','2024','--dry'],capture_output=True)
    # short run: 3 epochs, no periodic eval
    tmp=f'conf/smoke/bench_{tag}.yaml'
    s=open(cfg).read()
    s=re.sub(r'max\.epoch: \d+','max.epoch: 3',s)
    s=re.sub(r'eval\.every: \d+','eval.every: 999',s)
    s=re.sub(r'num\.max\.preepoch: \d+','num.max.preepoch: 2',s)
    open(tmp,'w').write(s)
    env=dict(os.environ,RUN_SEED='2024',RUN_TAG=f'BENCH_{tag}',PYTORCH_CUDA_ALLOC_CONF='')
    t0=time.time()
    r=subprocess.run([PY,'-c',f"""
import torch,sys,time
sys.argv=['main.py','--config','{tmp}']
torch.cuda.reset_peak_memory_stats()
t=time.time()
exec(open('main.py').read())
print('BENCH_WALL', time.time()-t)
print('BENCH_MEM', torch.cuda.max_memory_allocated()/1e9)
"""],capture_output=True,text=True,env=env)
    txt=r.stdout
    w=re.search(r'BENCH_WALL ([\d.]+)',txt); m=re.search(r'BENCH_MEM ([\d.]+)',txt)
    tp=re.search(r'trainable tensors=(\d+)',txt)
    out[name]={'wall_3ep_s':float(w.group(1)) if w else None,
               'peak_gpu_gb':float(m.group(1)) if m else None,
               'trainable_tensors':int(tp.group(1)) if tp else None}
    print(name,out[name],flush=True)
json.dump(out,open('results/wm/cost_benchmark.json','w'),indent=2)
print('saved results/wm/cost_benchmark.json')
