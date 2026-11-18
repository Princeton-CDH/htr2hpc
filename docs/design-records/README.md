# Design records

This folder holds design records for htr2hpc features: the decisions behind a feature and the plan for implementing it. They are written so colleagues can review a design before or during implementation.

## When to add one

Add a design record when a feature involves decisions worth reviewing or remembering, such as choices about data, interface, or behavior. Routine changes belong in commits and pull requests.

## File naming

Name each file after the GitHub issue it belongs to, not the pull request:

`<issue number>-<short-topic>.md`, e.g. `216-usage-reporting.md`

## Structure

- **Context:** the problem and why it needs solving.
- **Decisions:** what was decided for the feature as a whole, not just the latest round of changes.
- **Considered Options:** alternatives, where they help explain the decisions.
- **Planned changes:** what still needs to be implemented, if anything.

Update the record when decisions change, so it always describes the feature's current design.
