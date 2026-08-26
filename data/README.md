The scenario cache (`data/scenarios/`) is generated, not downloaded, and is
gitignored. Rebuild it with `make data`. There is no network access anywhere in
the default path.

An optional CSV loader (`sndsur.data.csvio`) reads a user-supplied network; see
`docs/METHOD.md`.
