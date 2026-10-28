# Minimal TUI

A full-screen, standard-library `curses` front end for the harness. It uses the
same `Chat` as `python -m harness`, so sessions, tools, context compression and
the system prompt are unchanged.

```bash
.venv/bin/python -m tui                   # new session
.venv/bin/python -m tui --session <id>    # resume
.venv/bin/python -m tui --db path.db      # another database
```

Keys:

- `Enter` send; `/help`, `/new`, `/sessions`, `/resume <id>`, `/usage`, `/exit`
  work as in the CLI.
- `Up`/`Down` walk the input history.
- `PgUp`/`PgDn` scroll the transcript; `Ctrl-G` jumps to the newest line.
- `Ctrl-C` quits.

The harness runs on a worker thread, so the spinner and scrolling stay alive
while the model is thinking. Output is plain text; markdown is not rendered.
POSIX terminal only.
