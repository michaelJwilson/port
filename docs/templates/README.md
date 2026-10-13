# `docs/templates`

Templates for documents this repository produces outside the code.

## `work-in-flight.html` (#335)

**One self-contained page that states what is open, in what order it merges,
and what is queued, result first.** It is published as a private claude.ai
artifact and refreshed hourly and whenever work lands.

| section | states | from |
| --- | --- | --- |
| header | snapshot time, a TL;DR, counts by state | the sections below |
| merge order | one lane per stack of PRs | each PR's `base`: a PR based on another PR's head stacks on it |
| open pull requests | title with its headline number, base, CI state, next action | open PRs, check runs on the head, mergeability |
| in progress and queued | status, the result first, then what remains | the task list |
| tickets filed | where each stands | issues |

**The rules it follows are `CLAUDE.md`'s.** The TL;DR opens the page, and
every row and card opens with its number, not its narrative (Writing
Style 8). A state is read from the check runs on the current head, never
inferred: `ok` is green, `wait` is pending or waiting on a maintainer to
approve a run, `bad` is red, conflicted, or needing a decision.

**To fill it:** copy the file, replace every `{FIELD}`, and repeat the
commented rows. No build step: the CSS is inline, colours are tokens on
`:root` with a dark set, and the only external resource is Google Fonts.
It stays a template rather than a generator, because the headline number
and the next action are judgements, not fields of an API.
