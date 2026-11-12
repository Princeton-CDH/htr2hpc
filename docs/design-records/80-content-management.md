# Content management for custom page content

Issue: [#80](https://github.com/Princeton-CDH/htr2hpc/issues/80) · PR: [#245](https://github.com/Princeton-CDH/htr2hpc/pull/245)

## Context

Custom page content (contact info, announcements, model descriptions, HPC setup instructions) is currently changed by overriding eScriptorium templates in htr2hpc. Every content update needs a code change and a deploy, and the content has to be written as raw HTML.

Staff should be able to edit this content themselves with a WYSIWYG editor, without touching HTML, and have it appear on specific pages.

How can htr2hpc let staff manage page content without template overrides for each page?

## Decisions

Staff edit content as Wagtail snippets in the Wagtail admin at `/cms/`, using Wagtail's rich text editor. Each snippet is shown in a fixed slot on one page, or on every page.

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
- `url_path` is compared with `request.path` exactly. The query string is not part of `request.path`, so `/contact/?x=1` matches `/contact/`. A missing trailing slash isn't a problem in practice: Django's `CommonMiddleware` (enabled in eScriptorium) redirects `/contact` to `/contact/` before the page is rendered.

### Where snippets display

- htr2hpc ships a **copy of eScriptorium's `base.html`** with two new blocks around `{% block body %}`, inside `<div id="main">`:

  - `pre_body` renders `before_main` snippets.
  - `post_body` renders `after_main` snippets.

  Every page that extends `base.html` gets the snippets. Individual page templates don't need to be overridden.

- When Wagtail is installed, settings put `htr2hpc/content/templates` first in `TEMPLATES["DIRS"]`, ahead of htr2hpc's and eScriptorium's template directories, so the copy overrides eScriptorium's `base.html`. Template `DIRS` are searched before app template folders, so the content app's templates would otherwise never be used.

- The copy has a comment at the top naming the eScriptorium version it was copied from (v1.0.1), and that it **must be updated when eScriptorium is upgraded**. `DEVELOPERNOTES.md` says the same.

- We will propose adding empty `pre_body` / `post_body` blocks upstream, so the copy can later be replaced by a small override.

- **Homepage heading:** `additional_heading.html` is an include eScriptorium provides for customizing the homepage jumbotron. It keeps the "eScriptorium" title and renders the `homepage_heading` snippet below it.

### New interface

eScriptorium v1.0 has two user interfaces: the legacy interface, and a new Vue-based interface (built with Vue 2). Each user chooses with the "Legacy mode" option on their profile (`User.legacy_mode`); eScriptorium doesn't give the new interface a separate name. Both interfaces render through the same Django `base.html`.

This was checked on a local dev instance (eScriptorium v1.0.1) by rendering pages with `legacy_mode` on and off, using a copy of `base.html` with marker blocks.

- `pre_body` and `post_body` render in both interfaces on every page checked: `/`, `/projects/`, `/profile/`, `/contact/`, `/models/`, `/quotas/instance/`.
- In the new interface, the blocks sit inside `<div id="main">`, outside the Vue mount points (e.g. `<div id="projects-list">`). Vue doesn't replace them, and they show above and below the page content.
- The homepage jumbotron (`additional_heading.html`) also displays in both interfaces.
- **Integration effort is low:** no changes are expected for the new interface. The only work is styling the snippet wrapper so it fits the new interface's dark theme.

### Profile page HPC instructions

The HPC setup instructions become editable, but only partly. They mix static text with parts that rich text can't render:

- the ssh command, which includes `{{ request.user.username }}` and `{{ current_site.domain }}`
- the two "HPC setup" buttons, which are forms with `{% csrf_token %}`

So the text moves into a `/profile/` `after_main` snippet, and the ssh command and setup button stay in the content app's `users/profile.html` override, rendered after the snippet.

### Without Wagtail

htr2hpc's own templates keep the `develop` versions of `additional_heading.html` and `users/profile.html`, so the homepage text and the HPC setup section still display when Wagtail isn't installed. The fallback profile template wraps the HPC section in an `hpc_setup` block, which the content app's override empties.

### Initial content

A fixture, `cdh_snippets`, preloads the current custom content as snippets:

- the homepage heading text, as a `homepage_heading` snippet
- the profile HPC instruction text, as a `/profile/` `after_main` snippet

After deploying, run `manage.py loaddata cdh_snippets` once.

### Optional dependency

- Wagtail is optional. Installing with `htr2hpc[all]` includes it. `all` also leaves room for other optional groups later.
- All Wagtail code lives in the `htr2hpc.content` app, which settings only add when Wagtail is installed. The context processor is registered only in that case, so it can import `ContentSnippet` at the top of the module.
- Only the Wagtail apps snippets need are installed: `wagtail.snippets`, `wagtail.admin` and `wagtail`. `wagtail.documents` isn't installed because its `Document` model clashes with eScriptorium's `core.Document` (both add `User.document_set`), which stops Django from starting. `wagtail.images` isn't needed either.
  - **Proposed revision:** this turned out to be wrong; the Wagtail admin doesn't work without `wagtail.users`, `wagtail.documents` and `wagtail.images`. See [Revision: Wagtail admin apps](#revision-wagtail-admin-apps) below.

### Tests

- CI runs the tests twice, with a build matrix: once without Wagtail and once with it (the `wagtail` extra). This makes sure Wagtail stays truly optional.
- Like `htr2hpc.settings`, `tests/test_settings.py` only adds the Wagtail apps and `htr2hpc.content` when Wagtail is installed.
- `tests/test_content.py` uses `pytest.importorskip("wagtail")`, so the content tests are skipped when Wagtail isn't installed. Coverage from both runs is combined.
- Without Wagtail, the content app isn't installed and `page_snippets` isn't registered, so there is nothing else to test there. The `ImportError` fallback test is removed along with the fallback.

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
   - Recreate the content migrations as a single initial migration. The model changes happened during development and were never deployed, so there's no migration history to keep.
2. **Templates** (`src/htr2hpc/content/templates/`)
   - Replace `base.html` with the full copy plus the `pre_body` / `post_body` blocks, with the upgrade comment at the top.
   - Update `additional_heading.html` to render the `homepage_heading` snippet.
   - Update `users/profile.html`: remove the instruction text and keep the ssh command and buttons.
   - Remove the overrides for `users/contactus.html` and `core/models_list/main.html`.
   - Keep the `develop` versions of `additional_heading.html` and `users/profile.html` in `src/htr2hpc/templates/` as the fallback without Wagtail.
   - In settings, put the content app's template directory first when Wagtail is installed.
   - Remove `wagtail.documents` and `wagtail.images` from the installed apps.
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

## Revision: Wagtail admin apps

Status: **proposed**, for review.

This revises the implementation merged in [#245](https://github.com/Princeton-CDH/htr2hpc/pull/245) and [#246](https://github.com/Princeton-CDH/htr2hpc/pull/246), as deployed to staging on 2026-11-12.

### Problem

After deploying to staging, the Wagtail admin errors as soon as a staff user logs in. It never worked: #245 didn't route the admin, so it was never opened, and the tests created snippets in code instead of through the admin.

Wagtail's admin can't run on the three apps we install (`wagtail.snippets`, `wagtail.admin`, `wagtail`). It also needs:

- `wagtail.users`: every admin page reads the logged-in user's Wagtail profile.
- `wagtail.documents` and `wagtail.images`: every admin page template looks up the documents and images API URLs (`wagtail_config` in `wagtailadmin_tags.py`, unchanged through Wagtail 7.2).

`wagtail.users` and `wagtail.images` can be installed as they are. `wagtail.documents` can't: its `Document` model clashes with eScriptorium's `core.Document`, which stops Django from starting.

### Why `wagtail.documents` clashes

Both models have a foreign key to the user model with no `related_name`, so Django gives both the same reverse names on `User`:

| Model | Field | Reverse accessor | Reverse query name |
|---|---|---|---|
| eScriptorium `core.Document` | `owner` | `document_set` | `document` |
| Wagtail `wagtaildocs.Document` | `uploaded_by_user` | `document_set` | `document` |

A custom Wagtail document model (`WAGTAILDOCS_DOCUMENT_MODEL`) doesn't avoid this: Wagtail still registers its built-in `Document` model whenever `wagtail.documents` is installed.

### Proposed fix: patch eScriptorium's `core.Document.owner`

Give `core.Document.owner` its own reverse names, so Wagtail's `Document` keeps the default ones. This follows the existing cdh-ansible patch that makes `User.email` nullable.

eScriptorium uses both reverse names itself, so three lines need patching, all in cdh-ansible's `escriptorium_setup` role:

| File | Line (v1.0.1) | Change |
|---|---|---|
| `apps/core/models.py` | `Document.owner` | add `related_name="escr_documents", related_query_name="escr_document"` |
| `apps/users/models.py` | `User.calc_disk_usage()` | `self.document_set` → `self.escr_documents` |
| `apps/reporting/views.py` | Leaderboard disk usage | `Sum('document__parts__...')` → `Sum('escr_document__parts__...')` |

These are the only uses in eScriptorium v1.0.1, outside migrations and tests.

- `core/models.py` has four identical `owner = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)` lines (on `Project`, `Document` and two others), so the patch has to match only the `Document` one: the `owner` line followed by `workflow_state`.
- Changing `related_name` doesn't change the database, but Django records it in model state, so htr2hpc needs a migration for it, like `0001_user_email_allow_null`.
- The names use an `escr_` prefix to make it clear they come from the patch, and to avoid eScriptorium's existing `shared_documents` reverse name on `Document`.

Then add `wagtail.users`, `wagtail.documents` and `wagtail.images` to the installed apps, in `htr2hpc.settings` and `tests/test_settings.py`.

### Prototype

I tested this locally against eScriptorium v1.0.1 and PostgreSQL 15, with the three Wagtail apps added:

- **Ansible patches:** ran the three `ansible.builtin.replace` tasks with Ansible on a clean eScriptorium checkout. Each matched once (only `Document.owner`, not the three other identical `owner` lines), and a second run changed nothing.
- **Migrations:** built the test database by running every migration from scratch, including a new htr2hpc migration recording the `related_name` (following `0001_user_email_allow_null`). Afterwards, `makemigrations --check` reports no pending changes for `core` or `content`.
- **Django checks:** no reverse accessor clash with the patch. Without it, the checks report the `fields.E304` clash, so a test catches it.
- **eScriptorium code using the old names:** `user.calc_disk_usage()` and the Leaderboard's disk usage query both work with the patched names. Wagtail's own `user.document_set` still exists.
- **Editing through the admin, as a logged-in staff user:**
  - `/cms/`, the snippets index, the content snippet list and the "add" page all return 200.
  - Creating a snippet through the admin form saves it, and it appears in the list.
  - Editing it changes the text, and unchecking "Live" hides it.
  - A snippet saved through the admin renders on its page.
  - Adding a second snippet for the same page and slot shows the form again with an error, not a server error.
- **Full test suite:** 197 passed, on a test database built from real migrations.

Not tested yet: a real deploy to staging with the patches.

### Planned changes

1. **cdh-ansible** (`roles/escriptorium_setup/tasks/main.yml`)
   - Add three `ansible.builtin.replace` tasks for the three lines above, next to the email patch.
   - Test each regex against eScriptorium v1.0.1 and check it matches exactly once.
2. **htr2hpc migration** (`src/htr2hpc/migrations/`)
   - Add a migration recording the new `related_name` on `core.Document.owner`, following `0001_user_email_allow_null`.
3. **htr2hpc settings** (`src/htr2hpc/settings.py`, `tests/test_settings.py`)
   - Add `wagtail.users`, `wagtail.documents` and `wagtail.images`.
4. **Tests** (`tests/test_content.py`)
   - Add tests that log in as a staff user and open the admin pages, and that add, edit and hide a snippet through the admin.
   - Add a test that runs Django's system checks, so a model clash fails the tests (pytest doesn't run the checks on its own).
   - Build the test database by running migrations instead of skipping them, so migrations are tested too. eScriptorium's `users.0015` migration needs `sites` migrated first.
   - Test patches: in CI, apply the same three edits to the eScriptorium checkout before running the tests, so the tests run against patched eScriptorium like staging.
5. **Docs**
   - Update this record's "Optional dependency" section.
   - In `DEVELOPERNOTES.md`, add the new patches to the list of things to check when upgrading eScriptorium.
6. **Upstream**
   - Propose adding `related_name` to `core.Document.owner` in eScriptorium, so the patch can be removed after an upgrade.

### Open questions

- Is patching eScriptorium's `core.Document` model acceptable, given it adds three patches to check on every eScriptorium upgrade?
- How should CI apply the patches: reuse the regexes from cdh-ansible, or keep a separate patch file in htr2hpc?

### Alternatives considered

1. **Custom Wagtail document model.** Rejected: Wagtail still registers its built-in `Document` model, so the clash stays.
2. **Placeholder documents API URL.** Leave `wagtail.documents` out and add an empty URL with the name the admin template looks up. Smaller, but relies on Wagtail internals and could break on a Wagtail upgrade.
3. **Django admin with a rich text editor instead of Wagtail.** No clash, but it reverses a reviewed decision and redoes part of #80.
