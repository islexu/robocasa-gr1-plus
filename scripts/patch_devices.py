"""Run LoTTA patch shards only on explicitly selected GPUs, including a single free GPU."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

def main():
    parser=argparse.ArgumentParser()
    for name in ('config','pairs','output','devices'): parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    devices=args.devices.split(',')
    if len(set(devices))!=len(devices) or any(not x.isdigit() for x in devices):
        raise ValueError('Choose distinct numeric GPU devices')
    pair_count=len(list(Path(args.pairs).glob('**/pairs/*.npz')))
    workers=min(pair_count,len(devices))
    if not workers: raise ValueError('No calibration pairs')
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    jobs=[]; logs=[]
    try:
        for rank,device in enumerate(devices[:workers]):
            env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=device; env['LOTTA_DEVICE']='cuda:0'
            log=open(output/f'rank-{rank}.log','a'); logs.append(log)
            command=[sys.executable,'-m','lotta.cli','patch','--config',args.config,'--pairs',args.pairs,
                     '--output',str(output/f'rank-{rank}'),'--shard',str(rank),'--shards',str(workers),'--resume']
            jobs.append(subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT))
        pending=set(range(workers))
        while pending:
            for i in list(pending):
                status=jobs[i].poll()
                if status is None: continue
                if status: raise RuntimeError(f'Patching worker {i} failed; inspect {output}/rank-{i}.log')
                pending.remove(i)
            if pending: time.sleep(.2)
    finally:
        for proc in jobs:
            if proc.poll() is None: proc.terminate()
        for proc in jobs:
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        for log in logs: log.close()

if __name__=='__main__': main()
