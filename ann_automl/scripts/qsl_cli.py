import os
import typing
import eel
import bottle
import pkg_resources
import qsl
import click

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
def start(jsonpath: str, targets: typing.List[str], batchSize: typing.Optional[int],
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

@click.group()
def cli():
    """The QSL CLI application."""

@click.command()
@click.argument("project", nargs=1)
@click.argument("targets", nargs=-1)
@click.option("-b", "--batch-size", "batchSize", default=None, type=int)
@click.option("-p", "--port", "port", default=8080, type=int)
@click.option("--host", "host", default="localhost", type=str)
def label(project, targets, batchSize, host, port):
    """Launch the labeling application."""
    if not project.endswith(".json"):
        click.echo(
            f"The project path must end in *.json. Received {project}.", err=True
        )
        return
    print(f"qsl label running at: http://{host}:{port}")
    start(jsonpath=project, targets=targets, batchSize=batchSize, host = host, port = port)


cli.add_command(label)

if __name__ == "__main__":
    cli()