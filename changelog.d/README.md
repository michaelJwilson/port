# `changelog.d`

Towncrier fragments. `pyproject.toml` points `[tool.towncrier]` here, and
`CLAUDE.md`'s **Documentation Sync** requires one per user-visible change.

## Writing one

One file per change, named `<issue>.<type>.md`, where `<issue>` is the ticket
or pull request number and `<type>` is one of `added`, `changed`, `fixed`,
`removed`, `security`. One sentence, in the past tense, describing what a
*user of the package* sees.

```
changelog.d/24.added.md
```

## What counts as user-visible

`port` is unusual here and it is worth saying once rather than deciding per
change. The package ships `python/port/` and the `oxiport` extension; almost
everything else this repository produces -- tests against a dependency,
audits, tickets, benchmark tables -- is not visible to anyone importing
`port`, however much it matters to the work.

**So a fragment is written when `python/port/` or the crate changes
behaviour, and not otherwise.** A new test, a new audit document, a raised
coverage floor and a `CLAUDE.md` edit are all changes to how this repository
works rather than to what it ships, and adding fragments for them would make
the changelog a second commit log.

Where that line is unclear, the test is whether someone who only ran
`import port` would notice.

## Retroactive fragments are not written

Nine pull requests merged before this directory existed, and none carries a
fragment. That is recorded rather than repaired: `CHANGELOG.md` does not yet
exist, `towncrier` has no release to build, and reconstructing fragments for
merged work would date them to the wrong release. The history is in the
commits and the pull requests.

The convention starts here.
