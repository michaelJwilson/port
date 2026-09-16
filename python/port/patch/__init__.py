"""Patch code for `cnaster`, written here because the dependency is read only.

`CLAUDE.md`: "Where one of them needs a change, report it here with the
evidence and leave landing it to that repository." A report is more useful
carrying the change it proposes, so each module here is a **drop-in
replacement** for one `cnaster` function -- same name, same signature but for
what the patch changes, no `port` imports -- so it can be copied across
without editing.

Every module states three things its docstring must carry, because
`CLAUDE.md` makes an optimization arrive with all of them: the function it
replaces, the referee that pins it equivalent, and the measured ratio at the
sizes the Measurement rule requires.
"""
