import os
import typing
import eel
import bottle
import pkg_resources
import qsl

class MediaLabeler(qsl.common.BaseMediaLabeler):
    def __init__(
        self,
        items=None,
        jsonpath=None,
        batchSize=None,
        host = "localhost",
        port = "8080"
    ):
        super().__init__(
            items=items,
            batchSize=batchSize,
            jsonpath=jsonpath,
            base={
                "url": f"http://{host}:{port}",
                "serverRoot": os.getcwd(),
            },
        )
        self.set_urls_and_type()
        eel.expose(self.init)
        eel.expose(self.sync)

    def init(self, key):
        return getattr(self, key)

    def sync(self, key, value):
        setattr(self, key, value)
        if key == "states":
            self.targets = [
                {**state, "target": target.get("target")}
                for target, state in zip(self.targets, self.states)
            ]
            self.set_buttons()
        if key == "action":
            if not value:
                return
            self.apply_action(value)

    def __setattr__(self, key, value):
        eel.sync(key, value)  # pylint: disable=no-member
        super().__setattr__(key, value)


# pylint: disable=unused-variable
def _start(jsonpath: str, targets: typing.List[str], batchSize: typing.Optional[int],
          host: str = "localhost", port: int = 8080):
    """Start Eel."""
    # A bit of a hack so that `qsl.files.build_url` works properly
    eel.BOTTLE_ROUTES = {
        "/files/<path:path>": (
            lambda path: bottle.static_file(path, root=os.getcwd()),
            {},
        ),
        **eel.BOTTLE_ROUTES,
    }
    eel.init(
        os.path.dirname(pkg_resources.resource_filename("qsl", "ui/eelapp/index.html")),
        [".js", ".html"],
    )
    labeler = MediaLabeler(
        items=[
            {"target": t, "type": qsl.files.guess_type(t)}
            for t in qsl.files.filepaths_from_patterns(targets)
        ],
        jsonpath=jsonpath,
        batchSize=batchSize,
        host = host,
        port = port
    )
    eel.start(
        "index.html",
        mode="chrome-app",
        host=host,
        port=port,
        size=(1280, 800),
    )

def start(jsonpath: str, host: str = "localhost", port: int = 8080):
    print(f"http://localhost:{port}")
    _start(jsonpath, targets = [], batchSize = None, host = host, port = port)

if __name__ == "__main__":
    labels_file = "/auto/projects/brain/ann-automl-gui/datasets/test1/tmpdir1/microtest/labels.json"
    start(labels_file, host = "0.0.0.0", port = 8080)