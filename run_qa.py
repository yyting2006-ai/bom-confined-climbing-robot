"""Persist test receipts and environment identity for the experiment release."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parent
    a.output.mkdir(parents=True,exist_ok=False)
    sources=sorted(root.glob('*.py'))+sorted((root/'src').glob('*.py'))+sorted((root/'tests').glob('*.py'))
    for path in sources:
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    checks=[]
    commands=[('unit_tests',[sys.executable,'-B','-X','utf8','-m','unittest','discover','-s',str(root/'tests'),'-p','test_*.py','-v']),
              ('dependencies',[sys.executable,'-m','pip','check'])]
    for name,command in commands:
        result=subprocess.run(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        output=a.output/f'{name}.log'
        output.write_text(result.stdout,encoding='utf-8')
        checks.append(dict(name=name,command=command,exit_code=result.returncode,
                           output_sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
    packages=sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()})
    (a.output/'requirements-lock.txt').write_text('\n'.join(packages)+'\n',encoding='utf-8')
    receipt=dict(utc=datetime.now(timezone.utc).isoformat(),python=sys.version,executable=sys.executable,
                 ast_parsed_files=len(sources),checks=checks,
                 source_hashes={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                 passed=all(x['exit_code']==0 for x in checks))
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=receipt['passed'],ast_parsed_files=len(sources),output=str(a.output))))
    if not receipt['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
