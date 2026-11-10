# Content management for custom page content

Issue: [#80](https://github.com/Princeton-CDH/htr2hpc/issues/80) · PR: [#245](https://github.com/Princeton-CDH/htr2hpc/pull/245)

## Context

Custom page content (contact info, announcements, model descriptions, HPC setup instructions) is currently changed by overriding eScriptorium templates in htr2hpc. Every content update needs a code change and a deploy, and the content has to be written as raw HTML.

Staff should be able to edit this content themselves with a WYSIWYG editor, without touching HTML, and have it appear on specific pages.

How can htr2hpc let staff manage page content without template overrides for each page?

## Decisions

Staff edit content as Wagtail snippets in the Wagtail admin, using Wagtail's rich text editor. Each snippet is shown in a fixed slot on one page, or on every page.

### Content model

- **`ContentSnippet`** has four fields:

  - `url_path`: the page to show it on, e.g. `/contact/`. Leave it blank to show it on all pages.
  - `slot`: where on the page it appears.
  - `body`: rich text, edited with Wagtail's WYSIWYG editor.
  - `live`: whether the snippet is published. Defaults to `True`, matching Wagtail's `Page` model. Unchecking it hides the snippet without deleting it.

- **Three slots**, chosen from a dropdown, so there are no typos:

  - `before_main`: "Before main content (any page)"
  - `after_main`: "After main content (any page)"
  - `homepage_heading`: "Homepage heading text". Shown in the homepage jumbotron, below the "eScriptorium" title.

  Announcements, contact info, model descriptions and the profile HPC instructions use `before_main` / `after_main` on the relevant page, or on all pages.

- **Help text** in the admin is written for staff, e.g. "URL for the page where this snippet should be displayed (e.g. /contact/). Leave blank to display on all pages."

- **One snippet per page and slot.** This is a database constraint: `UniqueConstraint` on `url_path` + `slot`.

  - It applies whether or not the snippet is live, so a page and slot can never have two snippets, even unpublished ones. There are no drafts.
  - Most content won't change once it's added; only occasional site-wide announcements are likely to change often. So one snippet per page and slot is enough.
  - A site-wide snippet (blank `url_path`) and a page-specific snippet in the same slot are different rows, so both are allowed and both display, site-wide first.

### Context processor

`htr2hpc.content.context_processors.page_snippets` adds the snippets for the current page to every template context.

- It queries live snippets whose `url_path` is the request path or blank: `ContentSnippet.objects.filter(live=True).filter(Q(url_path=request.path) | Q(url_path=""))`.
- It returns `page_snippets`, a dict mapping each slot name to a list of snippets, site-wide first. A slot with no snippets is simply missing from the dict, so templates render nothing.
- It runs one query per page request.
- `url_path` is compared with `request.path` exactly. The query string is not part of `request.path`, so `/contact/?x=1` matches `/contact/`, but a missing trailing slash doesn't match.

### Where snippets display

- htr2hpc ships a **copy of eScriptorium's `base.html`** with two new blocks around `{% block body %}`, inside `<div id="main">`:

  - `pre_body` renders `before_main` snippets.
  - `post_body` renders `after_main` snippets.

  Every page that extends `base.html` gets the snippets. Individual page templates don't need to be overridden.

- htr2hpc's template directory comes first in `TEMPLATES["DIRS"]`, so the copy overrides eScriptorium's `base.html`.

- The copy has a comment at the top naming the eScriptorium version it was copied from (v1.0.1), and that it **must be updated when eScriptorium is upgraded**. `DEVELOPERNOTES.md` says the same.

- We will propose adding empty `pre_body` / `post_body` blocks upstream, so the copy can later be replaced by a small override.

- **Homepage heading:** `additional_heading.html` is an include eScriptorium provides for customizing the homepage jumbotron. It keeps the "eScriptorium" title and renders the `homepage_heading` snippet below it.

### New interface

This was checked on a local dev instance (eScriptorium v1.0.1) by rendering pages with `legacy_mode` on and off, using a copy of `base.html` with marker blocks.

- `pre_body` and `post_body` render in both interfaces on every page checked: `/`, `/projects/`, `/profile/`, `/contact/`, `/models/`, `/quotas/instance/`.
- In the new interface, the blocks sit inside `<div id="main">`, outside the Vue mount points (e.g. `<div id="projects-list">`). Vue doesn't replace them, and they show above and below the page content.
- The homepage jumbotron (`additional_heading.html`) also displays in both interfaces.
- **Integration effort is low:** no changes are expected for the new interface. The only work is styling the snippet wrapper so it fits the new interface's dark theme.

### Profile page HPC instructions

The HPC setup instructions become editable, but only partly. They mix static text with parts that rich text can't render:

- the ssh command, which includes `{{ request.user.username }}` and `{{ current_site.domain }}`
- the two "HPC setup" buttons, which are forms with `{% csrf_token %}`

So the text moves into a `/profile/` `after_main` snippet, and the ssh command and buttons stay in the `users/profile.html` override.

### Initial content

A fixture, `cdh_snippets`, preloads the current custom content as snippets:

- the homepage heading text, as a `homepage_heading` snippet
- the profile HPC instruction text, as a `/profile/` `after_main` snippet

After deploying, run `manage.py loaddata cdh_snippets` once.

### Optional dependency

- Wagtail is optional. Installing with `htr2hpc[all]` includes it. `all` also leaves room for other optional groups later.
- All Wagtail code lives in the `htr2hpc.content` app, which settings only add when Wagtail is installed. The context processor is registered only in that case, so it can import `ContentSnippet` at the top of the module.

### Tests

- Wagtail is in the `dev` dependency group, and `tests/test_settings.py` always includes the Wagtail apps and `htr2hpc.content`. So the content tests always run in CI and locally.
- Without Wagtail, the content app isn't installed and `page_snippets` isn't registered, so there is nothing to test there. The `ImportError` fallback test is removed along with the fallback.
- Other htr2hpc tests don't depend on Wagtail.

## Considered Options

### Slots

1. Separate `intro`, `announcement`, `before_body` and `after_body` slots. Rejected: they overlap. `intro` and `before_body` are the same position, and the homepage showed both `intro` and `announcement`.
2. Two generic slots only, with the homepage heading and profile instructions left hard-coded. Rejected: that content is important and should be editable.
3. **Two generic slots plus one homepage slot (chosen).**

### Injecting snippets

1. Override `{% block body %}` in a `base.html` that extends eScriptorium's, and override each page template that needs `{{ block.super }}`. Rejected: upstream page templates don't call `block.super`, so every page would need its own override.
2. **A copy of the full base template with `pre_body` / `post_body` blocks (chosen).** It is a single file to maintain, and it is clearly marked for upgrades.

### Drafts

1. Several snippets per page and slot, with only one live at a time. Rejected: content rarely changes, so drafts aren't needed, and validating "only one live" is harder than a database constraint.
2. **One snippet per page and slot (chosen).**

## Planned changes

These changes revise the in-progress implementation in [#245](https://github.com/Princeton-CDH/htr2hpc/pull/245) as of `834e49e` (2026-11-04). That version has four overlapping slots (`intro`, `announcement`, `before_body`, `after_body`), an `is_active` field, and template overrides for each page that shows snippets.

1. **Model and admin** (`src/htr2hpc/content/models.py`, `wagtail_hooks.py`, new migration)
   - Replace the four slots with `before_main`, `after_main` and `homepage_heading`.
   - Rename `is_active` to `live`, defaulting to `True`.
   - Update the help text.
   - Add a migration for the slot and field changes.
2. **Templates** (`src/htr2hpc/content/templates/`)
   - Replace `base.html` with the full copy plus the `pre_body` / `post_body` blocks, with the upgrade comment at the top.
   - Update `additional_heading.html` to render the `homepage_heading` snippet.
   - Update `users/profile.html`: remove the instruction text and keep the ssh command and buttons.
   - Remove the overrides for `users/contactus.html` and `core/models_list/main.html`.
3. **Context processor** (`src/htr2hpc/content/context_processors.py`)
   - Filter on `live` instead of `is_active`.
   - Move the `ContentSnippet` import to the top and remove the `ImportError` fallback.
4. **Fixture and docs**
   - Rename `initial_snippets.json` to `cdh_snippets.json`, with the homepage heading and profile instruction snippets.
   - In `DEVELOPERNOTES.md`, change the install step to `htr2hpc[all]`, update the `loaddata` step to `cdh_snippets`, and document the base template copy.
5. **Tests** (`tests/test_content.py`)
   - Use the pytest-django `rf` fixture.
   - Test that two snippets with the same page and slot raise an `IntegrityError`.
   - Test that non-live snippets aren't returned.
   - Test that `pre_body` / `post_body` snippets render on a page that has no template override.
   - Remove the `ImportError` fallback test.
