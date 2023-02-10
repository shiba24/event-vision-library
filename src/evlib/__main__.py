"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Event Vision Library."""


if __name__ == "__main__":
    main(prog_name="event-vision-library")  # pragma: no cover
