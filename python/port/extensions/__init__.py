"""New functionality, with no `cnaster` counterpart to reproduce.

One of the four jobs `CLAUDE.md` gives a module. The rule for this directory
is that a module earns its place by being used and carries the narrowest
interface that works -- an extension is not a staging area for code that
might replace something one day. A module that *does* replace a named
`cnaster` function or class belongs under `patch/`, whatever it is called.

**Nothing here may appear in a swap row.** `tests/test_module_correspondence.py`
asserts it, because an extension installed over a `cnaster` name is a patch
that has not admitted to being one.
"""
