"""Command-line-interface to run SHARP model.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import click

from . import predict


@click.group()
def main_cli():
    """Run inference for SHARP model."""
    pass


main_cli.add_command(predict.predict_cli, "predict")

try:
    from . import render

    main_cli.add_command(render.render_cli, "render")
except (ImportError, OSError) as exc:
    render_import_error = str(exc)

    @click.command()
    def render():
        """Render a PLY file. Requires optional gsplat dependencies."""
        raise click.ClickException(
            f"Rendering requires optional gsplat dependencies: {render_import_error}"
        )

    main_cli.add_command(render, "render")
