"""Type stub for the compiled `port.oxiport` Rust extension.

Hand-written, so it can drift: keep the signatures here matching the
`#[pyfunction]` definitions in src/lib.rs. `mypy --strict` catches only the
direction where the stub is missing something a caller uses.
"""

def double(x: int) -> int: ...
