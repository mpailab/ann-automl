import datetime
import errno
import html
import json
import os
import random
import shlex
import textwrap
import time
import IPython
import subprocess
import tempfile
import psutil

from tensorboard import manager

explicit_qsl = os.environ.get("QSL_BINARY", None)
qsl_prog_name = "qsl" if explicit_qsl is None else explicit_qsl

_started = False

def start(args_string, timeout=datetime.timedelta(seconds=20)):
    """Launch a `qsl` instance as if at the command line.

    Args:
      args_string: Command-line arguments to qsl, to be
        interpreted by `shlex.split`: e.g., "--logdir ./logs --port 0".
        Shell metacharacters are not supported: e.g., "--logdir 2>&1" will
        point the logdir at the literal directory named "2>&1".
      timeout: `datetime.timedelta` object describing how long to wait for
        the subprocess to initialize a TensorBoard server and write its
        `TensorBoardInfo` file. If the info file is not written within
        this time period, `start` will assume that the subprocess is stuck
        in a bad state, and will give up on waiting for it. Note that in such 
        a case the subprocess will not be killed. Default value is 60 seconds.
    """
    global _started

    if _started:
        return

    print(f"Launching qsl {args_string}... ", end='', flush=True)
    _started = True

    parsed_args = shlex.split(args_string, comments=True, posix=True)
    #print("parsed_args =", parsed_args)
    cache_key = manager.cache_key(
        working_directory=os.getcwd(),
        arguments=parsed_args,
        configure_kwargs={},
    )
    _clear_infos(cache_key)

    (stdout_fd, stdout_path) = tempfile.mkstemp(prefix=".qsl-stdout-")
    (stderr_fd, stderr_path) = tempfile.mkstemp(prefix=".qsl-stderr-")
    start_time_seconds = time.time()
    try:
        process = subprocess.Popen(
            [qsl_prog_name] + parsed_args,
            stdout=stdout_fd,
            stderr=stderr_fd,
        )
    except OSError as e:
        the_qsl_binary = (
            "%r (set by the `QSL_BINARY` environment variable)" % qsl_prog_name
        )
        if e.errno == errno.ENOENT:
            message = (
                "ERROR: Could not find %s. Please ensure that your PATH contains "
                "an executable `qsl` program, or explicitly specify the path "
                "to a qsl binary by setting the `QSL_BINARY` "
                "environment variable." % the_qsl_binary
            )
        else:
            message = "ERROR: Failed to start %s: %s" % (the_qsl_binary, e)
        print('fail\n' + textwrap.fill(message))

    finally:
        os.close(stdout_fd)
        os.close(stderr_fd)

    poll_interval_seconds = 0.5
    end_time_seconds = start_time_seconds + timeout.total_seconds()
    while time.time() < end_time_seconds:
        time.sleep(poll_interval_seconds)
        subprocess_result = process.poll()
        if subprocess_result is not None:
            def format_stream(name, value):
                if value == "":
                    return ""
                elif value is None:
                    return "\n<could not read %s>" % name
                else:
                    return "\nContents of %s:\n%s" % (name, value.strip())

            message = (
                "ERROR: Failed to launch qsl (exited with %d).%s%s"
                % (
                    subprocess_result,
                    format_stream("stderr", _maybe_read_file(stderr_path)),
                    format_stream("stdout", _maybe_read_file(stdout_path)),
                )
            )
            print('fail\n' + message)
            break

        info = _find_matching_info(cache_key)
        if info is not None:
            _dump_info(info, stdout_path, stderr_path)
            break
    else:
        message = (
            "ERROR: Timed out waiting for qsl to start. "
            "It may still be running as pid %d." % process.pid
        )
        print('fail\n' + message)

    print('ok')

def _get_info_dir(owner="ann-automl"):
    return os.path.join(tempfile.gettempdir(), f".{owner}-info")

def _get_info_file(pid, owner="ann-automl"):
    return os.path.join(tempfile.gettempdir(), f".{owner}-info/pid-{pid}.info")

def _dump_info(info, stdout_path, stderr_path):
    dump = json.dumps({
        'pid': info.pid,
        'cache_key': info.cache_key,
        'stdout': stdout_path,
        'stderr': stderr_path
    })
    os.makedirs(_get_info_dir(), exist_ok=True)
    with open(_get_info_file(info.pid), "w") as outfile:
        outfile.write("%s\n" % dump)

def _clear_infos(cache_key):
    pids = set({})
    info_dir = _get_info_dir()
    if os.path.isdir(info_dir):
        for filename in os.listdir(info_dir):
            filepath = os.path.join(info_dir, filename)
            with open(filepath) as infile:
                values = json.loads(infile.read())
            if values['cache_key'] == cache_key:
                pids.add(values['pid'])
                os.remove(values['stdout'])
                os.remove(values['stderr'])
                os.remove(filepath)
                filepath = _get_info_file(values['pid'], owner="tensorboard")
                if os.path.isfile(filepath):
                    os.remove(filepath)
    for proc in psutil.process_iter():
        if proc.name() == qsl_prog_name and proc.pid in pids:
            proc.kill()


def _find_matching_info(cache_key):
    """Find a running qsl instance compatible with the cache key.

    Returns:
      A `qsl` object, or `None` if none matches the cache key.
    """
    infos = [i for i in manager.get_all() if i.cache_key == cache_key]
    info = max(infos, key=lambda x: x.start_time) if infos else None
    return info


def _maybe_read_file(filename):
    """Read the given file, if it exists.

    Args:
      filename: A path to a file.

    Returns:
      A string containing the file contents, or `None` if the file does
      not exist.
    """
    try:
        with open(filename) as infile:
            return infile.read()
    except IOError as e:
        if e.errno == errno.ENOENT:
            return None
