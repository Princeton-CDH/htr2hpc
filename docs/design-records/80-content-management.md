# Content management for custom page content

## Context

Custom page content (contact info, announcements, model descriptions) is currently changed by overriding eScriptorium templates in htr2hpc, so every content update needs a code change and a deploy. Staff should be able to edit this content themselves and have it appear on specific pages.

How can htr2hpc let staff manage page content without template overrides for each page?

## Decisions

Staff edit content as Wagtail snippets in the Wagtail admin. Each snippet is shown before or after the main content of one page, or of every page.

### Content model

- **`ContentSnippet`** has four fields:

  - `url_path`: the page to show it on, e.g. `/contact/`. Leave it blank to show it on all pages.
  - `slot`: where on the page it appears.
  - `body`: rich text.
  - `is_active`: unpublish without deleting.

- **Two slots**, chosen from a dropdown, so there are no typos:

  - `before_main`: "Before main content (any page)"
  - `after_main`: "After main content (any page)"

  Announcements, contact info and model descriptions all use these two slots on the relevant page, or on all pages.

- **Help text** in the admin is written for staff, e.g. "URL for the page where this snippet should be displayed (e.g. /contact/). Leave blank to display on all pages."

- **One snippet per page and slot** (unique `url_path` + `slot`). A site-wide snippet and a page-specific snippet in the same slot both display, site-wide first.

### Where snippets display

- htr2hpc ships a **copy of eScriptorium's `base.html`** with two new blocks around `{% block body %}`:

  - `pre_body` renders `before_main` snippets.
  - `post_body` renders `after_main` snippets.

  Every page that extends `base.html` gets the snippets. Individual page templates don't need to be overridden.

- The copy has a comment at the top naming the eScriptorium version it was copied from, and that it **must be updated when eScriptorium is upgraded**. `DEVELOPERNOTES.md` says the same.

- We will propose adding empty `pre_body` / `post_body` blocks upstream, so the copy can later be replaced by a small override.

- **New interface:** no extra work is expected. New-interface pages (e.g. `core/project_list.html`) also extend `base.html` and render inside `{% block body %}`, switching on `legacy_mode` within it. Blocks placed outside `body` therefore appear in both interfaces. Layout and styling in the new interface still need checking on a dev instance.

### Content that stays in templates

- The homepage jumbotron text (`additional_heading.html`) and the HPC setup instructions on the profile page stay as htr2hpc template text, as they are on `develop`. Both sit in the middle of a page layout that a before/after slot can't reach, and they rarely change.
- **No initial fixture.** Loading content with `loaddata` would work well, but no existing content moves into snippets, so there is nothing to preload. The `initial_snippets` fixture is removed rather than renamed.
- **Open question:** should the homepage text be editable too? If so, we would add a third slot, `homepage_heading`, rendered in `additional_heading.html`. eScriptorium already provides that include for this purpose, so it needs no extra template override. The fixture would then come back as `cdh_snippets` with that text preloaded.

### Optional dependency

- Wagtail is optional. Installing with `htr2hpc[all]` includes it. `all` also leaves room for other optional groups later.
- All Wagtail code lives in the `htr2hpc.content` app, which settings only add when Wagtail is installed. The context processor is registered only in that case, so it can import `ContentSnippet` at the top of the module.

## Considered Options

### Slots

1. Separate `intro`, `announcement`, `before_body` and `after_body` slots. Rejected: they overlap. `intro` and `before_body` are the same position, and the homepage showed both `intro` and `announcement`.
2. **Two generic slots, before and after the main content (chosen).**

### Injecting snippets

1. Override `{% block body %}` in a `base.html` that extends eScriptorium's, and override each page template that needs `{{ block.super }}`. Rejected: upstream page templates don't call `block.super`, so every page would need its own override.
2. **A copy of the full base template with `pre_body` / `post_body` blocks (chosen).** It is a single file to maintain, and it is clearly marked for upgrades.

## Planned changes

1. **Model and admin** (`src/htr2hpc/content/models.py`, migration)
   - Replace the four slots with `before_main` / `after_main`.
   - Update the help text.
2. **Templates** (`src/htr2hpc/content/templates/`)
   - Replace `base.html` with the full copy plus the `pre_body` / `post_body` blocks.
   - Remove the overrides for `additional_heading.html`, `users/profile.html`, `users/contactus.html` and `core/models_list/main.html`. Restore the `develop` versions of `additional_heading.html` and `users/profile.html` in `src/htr2hpc/templates/`.
3. **Context processor** (`src/htr2hpc/content/context_processors.py`)
   - Move the `ContentSnippet` import to the top and remove the `ImportError` fallback.
4. **Fixture and docs**
   - Remove `initial_snippets.json`. If the homepage slot is added, replace it with `cdh_snippets.json` instead.
   - In `DEVELOPERNOTES.md`, change the install step to `htr2hpc[all]`, remove the `loaddata` step, and document the base template copy.
5. **Tests** (`tests/test_content.py`)
   - Use the pytest-django `rf` fixture.
   - Test that snippets render before and after the main content on a page that has no template override.

### Questions for acceptance testing

- Is one snippet per page and slot enough, or do we need several (e.g. drafts) with only one active at a time?
- Do snippets look right in the new interface as well as the legacy one?
