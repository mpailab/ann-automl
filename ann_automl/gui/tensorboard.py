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

from tensorboard import manager

_is_started = False
_port = None
_pid = None

def start(args_string):
    """Launch a TensorBoard instance as if at the command line.

    Args:
      args_string: Command-line arguments to TensorBoard, to be
        interpreted by `shlex.split`: e.g., "--logdir ./logs --port 0".
        Shell metacharacters are not supported: e.g., "--logdir 2>&1" will
        point the logdir at the literal directory named "2>&1".
    """
    global _is_started, _port, _pid

    if _is_started:
        print("Warning: TensorBoard is already launched")

    print("Launching TensorBoard ...")
    _is_started = True

    parsed_args = shlex.split(args_string, comments=True, posix=True)
    start_result = manager.start(parsed_args)

    if isinstance(start_result, manager.StartLaunched):
        _port = start_result.info.port
        _pid = start_result.info.pid

    elif isinstance(start_result, manager.StartReused):
        template = (
            "Reusing TensorBoard on port {port} (pid {pid}), started {delta} ago. "
            "(Use '!kill {pid}' to kill it.)"
        )
        message = template.format(
            port=start_result.info.port,
            pid=start_result.info.pid,
            delta=_time_delta_from_info(start_result.info),
        )
        print(message)
        _port = start_result.info.port
        _pid = start_result.info.pid

    elif isinstance(start_result, manager.StartFailed):

        def format_stream(name, value):
            if value == "":
                return ""
            elif value is None:
                return "\n<could not read %s>" % name
            else:
                return "\nContents of %s:\n%s" % (name, value.strip())

        message = (
            "ERROR: Failed to launch TensorBoard (exited with %d).%s%s"
            % (
                start_result.exit_code,
                format_stream("stderr", start_result.stderr),
                format_stream("stdout", start_result.stdout),
            )
        )
        print(message)

    elif isinstance(start_result, manager.StartExecFailed):
        the_tensorboard_binary = (
            "%r (set by the `TENSORBOARD_BINARY` environment variable)"
            % (start_result.explicit_binary,)
            if start_result.explicit_binary is not None
            else "`tensorboard`"
        )
        if start_result.os_error.errno == errno.ENOENT:
            message = (
                "ERROR: Could not find %s. Please ensure that your PATH contains "
                "an executable `tensorboard` program, or explicitly specify the path "
                "to a TensorBoard binary by setting the `TENSORBOARD_BINARY` "
                "environment variable." % (the_tensorboard_binary,)
            )
        else:
            message = "ERROR: Failed to start %s: %s" % (
                the_tensorboard_binary,
                start_result.os_error,
            )
        print(textwrap.fill(message))

    elif isinstance(start_result, manager.StartTimedOut):
        message = (
            "ERROR: Timed out waiting for TensorBoard to start. "
            "It may still be running as pid %d." % start_result.pid
        )
        print(message)

    else:
        raise TypeError(
            "Unexpected result from `manager.start`: %r.\n"
            "This is a TensorBoard bug; please report it." % start_result
        )

def _time_delta_from_info(info):
    """Format the elapsed time for the given TensorBoardInfo.

    Args:
      info: A TensorBoardInfo value.

    Returns:
      A human-readable string describing the time since the server
      described by `info` started: e.g., "2 days, 0:48:58".
    """
    delta_seconds = int(time.time()) - info.start_time
    return str(datetime.timedelta(seconds=delta_seconds))


def interface(port=None, height=None, print_message=False):
    """Interface to display a TensorBoard instance already running on this machine.

    Args:
      port: The port on which the TensorBoard server is listening, as an
        `int`, or `None` to automatically select the most recently
        launched TensorBoard.
      height: The height of the frame into which to render the TensorBoard
        UI, as an `int` number of pixels, or `None` to use a default value
        (currently 800).
      print_message: True to print which TensorBoard instance was selected
        for display (if applicable), or False otherwise.
    """
    port = port or _port
    if height is None:
        height = 800

    if port is None:
        infos = manager.get_all()
        if not infos:
            raise ValueError(
                "Can't display TensorBoard: no known instances running."
            )
        else:
            info = max(manager.get_all(), key=lambda x: x.start_time)
            port = info.port
    else:
        infos = [i for i in manager.get_all() if i.port == port]
        info = max(infos, key=lambda x: x.start_time) if infos else None

    if print_message:
        if info is not None:
            message = (
                "Selecting TensorBoard with {data_source} "
                "(started {delta} ago; port {port}, pid {pid})."
            ).format(
                data_source=manager.data_source_from_info(info),
                delta=_time_delta_from_info(info),
                port=info.port,
                pid=info.pid,
            )
            print(message)
        else:
            # The user explicitly provided a port, and we don't have any
            # additional information. There's nothing useful to say.
            pass

    return _interface(port=port, height=height)


def _interface(port, height):
    """Internal version of `interface`.

    Args:
      port: As with `interface`.
      height: As with `interface`.
    """
    import IPython.display

    frame_id = "tensorboard-frame-{:08x}".format(random.getrandbits(64))
    shell = """
      <iframe id="%HTML_ID%" width="100%" height="%HEIGHT%" frameborder="0">
      </iframe>
      <script>
        (function() {
          const frame = document.getElementById(%JSON_ID%);
          const url = new URL(%URL%, window.location);
          const port = %PORT%;
          if (port) {
            url.port = port;
          }
          frame.src = url;
        })();
      </script>
    """
    proxy_url = os.environ.get("TENSORBOARD_PROXY_URL")
    if proxy_url is not None:
        # Allow %PORT% in $TENSORBOARD_PROXY_URL
        proxy_url = proxy_url.replace("%PORT%", "%d" % port)
        replacements = [
            ("%HTML_ID%", html.escape(frame_id, quote=True)),
            ("%JSON_ID%", json.dumps(frame_id)),
            ("%HEIGHT%", "%d" % height),
            ("%PORT%", "0"),
            ("%URL%", json.dumps(proxy_url)),
        ]
    else:
        replacements = [
            ("%HTML_ID%", html.escape(frame_id, quote=True)),
            ("%JSON_ID%", json.dumps(frame_id)),
            ("%HEIGHT%", "%d" % height),
            ("%PORT%", "%d" % port),
            ("%URL%", json.dumps("/")),
        ]

    for (k, v) in replacements:
        shell = shell.replace(k, v)
    return IPython.display.HTML(shell)
