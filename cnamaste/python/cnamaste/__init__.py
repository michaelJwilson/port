"""`cnaster` at the lock's pin, renamed, with `port`'s default patches folded in (#392).

The pin has no `__init__.py`, so `cnaster` imports as a namespace package. This
one makes `cnamaste` a regular package, which is what stops a directory named
`cnamaste` elsewhere on the path -- the repository root, for one -- from
joining it. It imports nothing: each module is loaded where it is used.
"""
