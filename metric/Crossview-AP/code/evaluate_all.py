import subprocess

config_dir = ".../Crossview-AP/expt/sample"
langs = ['kr', 'en', 'zh', 'jp']

for lang in langs:
    print(lang.upper())
    cmd = f"python evaluate.py --config {config_dir}/config_{lang}_10_eval.json --result_dir .../Crossview-AP/datasets/result"
    subprocess.run(cmd, shell=True, check=True)
    