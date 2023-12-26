import shlex
import subprocess

def launch(labels_file, host="localhost", port=8080):
    qsl_cpi = "ann_automl/scripts/qsl_cli.py"
    proc = simple_start(f"{qsl_cpi} label {labels_file} --port {port} --host {host} 2>/dev/null")
    return proc

def simple_start(args_string):
    parsed_args = shlex.split(args_string, comments=True, posix=True)
    process = subprocess.Popen(
            ["python3"] + parsed_args
        )
    return process