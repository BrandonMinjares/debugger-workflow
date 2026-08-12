import click
from click.testing import CliRunner


def test_with_echo_via_pager() -> None:
    @click.command()
    def cli() -> None:
        click.echo_via_pager("Hello, Click!")

    result = CliRunner().invoke(cli)

    assert result.exception is None
    assert result.output == "Hello, Click!\n"
