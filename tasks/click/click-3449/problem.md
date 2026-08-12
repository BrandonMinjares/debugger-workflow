# I/O operation on closed file with `CliRunner` and `echo_via_pager`

When `echo_via_pager` is used in a command, `CliRunner.invoke` fails with:

```text
ValueError: I/O operation on closed file.
```

This occurs with Click 8.4 but not Click 8.3.3.

```python
from click import command, echo_via_pager
from click.testing import CliRunner


@command()
def cli():
    echo_via_pager("Hello, Click!")


runner = CliRunner()
runner.invoke(cli)
```
