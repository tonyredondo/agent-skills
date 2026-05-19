---
name: code-diff-walkthrough
description: Generate bilingual annotated HTML walkthroughs for pull requests, git diffs, feature branches, or implementation changes. Use when the user asks for a PR walkthrough, an informed code review aid, an explanation of why each diff block exists, or a shareable English/Spanish HTML artifact after implementing a plan.
---

# Code Diff Walkthrough

## Workflow

Use `scripts/generate_walkthrough.py` to produce a self-contained diff review artifact. The script reads the real git diff, groups it by file and hunk, and writes:

- `index.html` in English
- `index.es.html` in Spanish
- `pr.diff`
- `diff-stat.txt`
- `commits.txt`
- `manifest.json`

Default output goes to a temporary directory outside the repository. Prefer that default unless the user explicitly requests a location.

Each hunk should show the nearest detected code context, such as a class or method, and include buttons for loading extra unchanged lines before or after the hunk. When a hunk's annotation column is taller than the initially visible code, automatically reveal enough available surrounding lines to use the vertical space instead of leaving a blank code pane. Keep any remaining surrounding lines available through the manual before/after buttons. Keep horizontal scrolling inside the code pane so the annotation column and UI controls remain visible.

Hunk explanations should be proportional to the actual diff. Base automatic notes on added and removed lines, not on surrounding context, and avoid broad assertions for tiny mechanical changes. Small hunks should usually get one concrete note, especially for simple assignments, configuration propagation, imports, or wiring. The right-hand hunk section is titled as an explanation of why the block changed, so generated notes must describe the purpose of the change, not reviewer checklist advice such as "check that..." or "conviene revisar...".

Diff code should include syntax highlighting through a pinned CDN dependency, with a plain escaped-code fallback if the dependency cannot load. Highlight at least the common review languages used across Datadog repos, including C#, Go, Java, JavaScript, TypeScript, Python, Ruby, Rust, C/C++, F#, Kotlin, Swift, SQL, JSON, YAML, TOML/INI, XML/HTML/MSBuild-style project files, shell, PowerShell, CSS, Markdown, Dockerfiles, and Makefiles. The highlighting must preserve whitespace, line selection, diff colors, and copyable plain text from the browser.

The generated HTML also supports selecting diff lines for follow-up review. Click a line, then click another line in the same file to select a range. Insert an inline comment composer directly below the selected range, not as a permanent per-file header block. The composer should behave like a lightweight GitHub review comment box: cancel or save the comment, persist saved comments in `localStorage`, render saved comments back in read-only mode near their selected lines, and allow editing or deleting each saved comment.

Saved review comments must support multiple comments across multiple ranges and files. Add Files-tab-level actions to copy all saved comments to the clipboard as a review bundle containing each `path:start:end` reference plus its comment text, and to clear all saved comments from `localStorage`.

The generated HTML also supports incremental review tracking. Each file can be marked reviewed or pending; that state is stored in browser `localStorage` using a stable review key and per-file diff fingerprints, so unchanged files stay reviewed across regenerated artifacts while changed files return to pending.

The generated HTML should also remember which file diffs the reviewer expanded. Store the expanded-file set in browser `localStorage` using the same stable review key plus each file's diff fingerprint, so unchanged files reopen on reload or regenerated artifacts while changed files return to collapsed. The per-file chevron, "Expand all", and "Collapse all" controls must update that state.

For long files, the file header and review controls should stay sticky while scrolling through that file, then scroll away naturally when the next file takes over. Selection actions should stay contextual to the selected lines.

The generated page should use a professional light review interface: compact top navigation, a persistent file queue sidebar, readable diff panels, clear addition/deletion counters in each file header, and architecture/glossary cards that remain useful without overwhelming the code review flow. File diffs should start collapsed by default so reviewers can open them one by one. Collapsed files should shrink to their header row only, with status and change counters still visible. Keep the sidebar visually distinct from the diff pane with a restrained tinted background and small internal panels for search, progress, actions, and the file queue; the main diff area should remain lighter.

Split the page into three primary tabs: "How to read this walkthrough", "Architecture", and "Files". Each tab must own an independent scroll container, so a reviewer can inspect architecture, return to the file review tab, and keep the same code scroll position. The file queue sidebar belongs to the Files tab and must list only real diff files, not synthetic navigation entries such as Architecture.

The top header should keep the main title visually distinct with restrained color, keep range/build metadata aligned to the right on desktop, and center the primary section links below the title row. On narrow screens, allow those links to return to left alignment so horizontal scrolling remains predictable.

File panels and architecture panels should share the same restrained 8px corner radius. Shadows should be subtle and close to the surface, not heavy floating-card shadows.

File expand/collapse should use an icon-only chevron before the file title, pointing right when collapsed and down when open. The file header row should also toggle the file when clicked, while clicks on header controls such as review buttons must keep their own behavior and not also expand or collapse the file. Keep review status and change counters on the right side of the header, and do not duplicate expansion as a text button in the header actions.

When a reviewer collapses an individual file from its sticky header while scrolled deep inside that file, immediately scroll the Files tab back to that file's header. The closed file should end up visible at the top of the review pane so the reviewer can move naturally to the next file. Do not apply this jump for initial state restoration or bulk "Collapse all".

The file queue sidebar should be collapsible from an in-page control. When collapsed on desktop, the main review pane should reclaim the full available width, leaving only a compact control to reopen navigation. Preserve the user's sidebar state in browser `localStorage` for regenerated artifacts with the same review key.

Architecture edge labels should remain readable even in dense diagrams. Render them as compact interactive labels with full edge details available through hover/click, rather than relying only on tiny SVG text on top of connector lines. Place labels with collision awareness so they prefer free space near their own connector and do not sit on top of node cards or other edge labels when avoidable. If a label must be offset away from the connector, render a small anchor/leader line back to the exact connector point so the label's owner edge remains obvious.

Architecture graph spacing should leave enough horizontal and vertical room for edge labels to sit close to their connector. Prefer a wider scrollable diagram over a compact graph where labels are forced far away from the lines they describe.

Each architecture flow panel should include local zoom controls with `+` and `-` buttons inside the graph panel they affect. Zoom must be visual scaling of that graph, not a re-layout of node columns, gaps, padding, or label placement. Preserve scrollability, redraw edges/labels after changes, and persist per flow in `localStorage` for regenerated artifacts with the same review key.
Only the zoom buttons should change zoom. Do not attach zoom handlers to the graph canvas, flow grid, nodes, labels, or wrapper elements, and keep internal state attributes distinct from button action attributes.

## Quick Start

From a repository checkout:

```bash
python3 /Users/tony.redondo/.codex/skills/code-diff-walkthrough/scripts/generate_walkthrough.py --repo "$PWD" --base master --head HEAD
```

For a GitHub PR in the current checkout:

```bash
python3 /Users/tony.redondo/.codex/skills/code-diff-walkthrough/scripts/generate_walkthrough.py --repo "$PWD" --pr 1234
```

With implementation context:

```bash
python3 /Users/tony.redondo/.codex/skills/code-diff-walkthrough/scripts/generate_walkthrough.py --repo "$PWD" --base master --head HEAD --context docs/path/to/plan.md
```

## Review Quality Rules

Before returning, verify that:

- the generated HTML files exist;
- both English and Spanish files parse their embedded JSON;
- there are no external scripts or stylesheets except the approved pinned highlight.js CDN assets;
- selecting lines inserts an inline comment composer with a visible `path:start:end` reference;
- syntax highlighting is present in generated diff code and uses only the approved pinned CDN dependency;
- marking a file reviewed updates the sidebar/progress UI and persists through `localStorage`;
- expanded file state persists through `localStorage`, survives reload for unchanged files, and is updated by per-file, expand-all, and collapse-all controls;
- saved review comments persist through `localStorage`, render beside their selected line ranges, can be edited/deleted, and can be copied or cleared together from the Files tab;
- the three primary tabs switch without a full-page scroll jump, and each tab preserves its own scroll position while switching;
- architecture edge labels do not overlap node cards or each other in the generated browser view;
- architecture flow zoom controls can zoom in and out on multiple flow panels without breaking edge rendering or label placement;
- tall annotation blocks automatically reveal available surrounding code lines to reduce unused blank space while preserving the manual context buttons;
- long-file review controls are inside the sticky file header wrapper;
- the file count in the manifest matches the diff;
- the output path is outside the repo unless the user asked otherwise.

If the user provided a plan or you have implementation context, pass it with `--context`. If the generated comments need project-specific nuance, use `--notes-json` with file or regex notes instead of manually editing the HTML.

## Notes JSON Workflow

For non-trivial PRs, create a temporary `notes-json` file before generating the final walkthrough. Treat this file as the explanation layer for the current diff. The Python generator should render and map notes; it should not be taught PR-specific facts in code.

Use this process:

1. Inspect the real diff first.
   - Read `git diff --stat`, the changed file list, and the most important implementation hunks.
   - Identify file roles from the actual code, not from filename keywords alone.
   - For lifecycle-sensitive flows, inspect both the producer and consumer code before writing a note.

2. Draft notes in an external JSON file outside the repo, usually under `/tmp`.
   - Use `files` for file-level rationale.
   - Use `patterns` for hunk-level explanations that should match specific changed lines.
   - Use `architecture` for call/data-flow diagrams and type glossary when the PR has several interacting parts.
   - Keep PR-specific facts in this JSON, not in `generate_walkthrough.py`.

3. Make hunk notes explain only the changed block visible in the hunk.
   - A `using`, `import`, package declaration, or generated metadata entry should not inherit the whole file rationale.
   - Tiny mechanical hunks should say that they are mechanical when that is the truth.
   - Do not claim runtime behavior for metadata, imports, formatting, or descriptor-only changes.
   - Do not say “this changes coverage/reporting behavior” unless the shown lines actually do that.
   - If the hunk is only plumbing for later code, phrase it narrowly, for example: “This hunk only brings X into scope; the behavior change appears in later code hunks.”

4. Prefer precise pattern mappings over broad keyword mappings.
   - Good: a regex that matches `RecordActualItrSkip`, `<type fullname="System.OverflowException" />`, or `CIVisibilityItrCoverageBackfillCommand`.
   - Bad: a regex that matches every `xml`, `coverage`, `test`, or `Exception` line and produces a broad explanation.
   - If a broad regex is needed, verify every affected hunk still makes sense.

5. Review the notes before returning the artifact.
   - Parse the generated HTML's embedded JSON.
   - Print or inspect hunk notes for representative files, especially tiny hunks, XML/config files, tests, and new abstractions.
   - Check screenshots or browser views for a few expanded files when the user is likely to read the artifact visually.
   - If a note sounds like it was copied from the file rationale instead of written for the hunk, fix the JSON or the generic note-selection behavior and regenerate.

6. Iterate until the notes are defensible.
   - The final walkthrough should not require the reviewer to trust hidden context for each block.
   - File rationale may describe the file’s role; hunk notes must describe the specific block.
   - If a hunk cannot be explained precisely from the visible diff, use a neutral note rather than inventing intent.

The `files` map may use either a string or a localized object. A localized object can also override the category when filename-based categorization is misleading:

```json
{
  "files": {
    "path/to/file.xml": {
      "category": {
        "en": "Trimming/linker metadata",
        "es": "Metadata de trimming/linker"
      },
      "en": "This file-level rationale explains the role of the file.",
      "es": "Esta explicación de fichero describe el rol del fichero."
    }
  },
  "patterns": [
    {
      "regex": "<type fullname=\"System\\.OverflowException\"\\s*/>",
      "en": "This hunk-level note explains only the matched block.",
      "es": "Esta nota de hunk explica solo el bloque matcheado."
    }
  ]
}
```

For complex PRs, `--notes-json` can also include an `architecture` section. Use it when the diff introduces several new types, cross-process wiring, request/response flows, lifecycle-sensitive behavior, or multiple adapters. The generated HTML will render self-contained visual call/data-flow diagrams and a type glossary. Each node or glossary item can list related files; clicking it filters the walkthrough to those files.

Architecture diagrams must be grounded in the real implementation, not just inferred from type names. Before adding or updating an architecture section, inspect the actual caller and callee code paths. If a method is only called behind a guard, put that guard in `when` or on the edge label. Do not merge legacy and new behavior into one node when the code has separate branches. Add `evidence` references such as `src/Foo.cs:123` for nodes that represent control-flow decisions, lifecycle behavior, or safety gates.

Example:

```json
{
  "architecture": {
    "title": {
      "en": "Architecture",
      "es": "Arquitectura"
    },
    "summary": {
      "en": "How the changed pieces call into each other.",
      "es": "Como se llaman entre si las piezas modificadas."
    },
    "sections": [
      {
        "id": "request-flow",
        "title": {
          "en": "Request flow",
          "es": "Flujo de request"
        },
        "summary": {
          "en": "The high-level call path for the new behavior.",
          "es": "El arbol de llamadas principal del nuevo comportamiento."
        },
        "nodes": [
          {
            "id": "entrypoint",
            "label": "Entrypoint.Method",
            "detail": {
              "en": "Starts the feature flow.",
              "es": "Inicia el flujo de la feature."
            },
            "when": {
              "en": "The feature entrypoint is invoked.",
              "es": "Cuando se invoca el punto de entrada de la feature."
            },
            "evidence": ["src/Entrypoint.cs:42"],
            "kind": "process",
            "files": ["src/Entrypoint.cs"],
            "column": 1,
            "row": 1
          },
          {
            "id": "backend",
            "label": "BackendClient.GetAsync",
            "detail": {
              "en": "Fetches the backend contract.",
              "es": "Obtiene el contrato del backend."
            },
            "when": {
              "en": "Only after the entrypoint guard allows the backend request.",
              "es": "Solo despues de que el guard del entrypoint permite el request al backend."
            },
            "evidence": [
              {
                "file": "src/BackendClient.cs",
                "line": 88,
                "label": {
                  "en": "Backend request call",
                  "es": "Llamada al request backend"
                }
              }
            ],
            "kind": "external",
            "files": ["src/BackendClient.cs"],
            "column": 2,
            "row": 1
          }
        ],
        "edges": [
          {
            "from": "entrypoint",
            "to": "backend",
            "label": {
              "en": "requests",
              "es": "consulta"
            },
            "when": {
              "en": "Only on the backend-enabled branch.",
              "es": "Solo en la rama donde el backend esta habilitado."
            },
            "evidence": ["src/Entrypoint.cs:45"]
          }
        ]
      }
    ],
    "glossary": [
      {
        "term": "BackendClient",
        "description": {
          "en": "Owns the remote request and response parsing.",
          "es": "Gestiona el request remoto y el parseo de la respuesta."
        },
        "evidence": ["src/BackendClient.cs:88"],
        "files": ["src/BackendClient.cs"]
      }
    ]
  }
}
```

## Script Options

Useful options:

- `--repo PATH`: repository checkout. Defaults to current directory.
- `--pr NUMBER_OR_URL`: use GitHub PR metadata through `gh`.
- `--base REF --head REF`: compare a branch/range with `git diff base...head`.
- `--range RANGE`: pass an exact git diff range.
- `--context PATH`: include plan or design context. Can be repeated.
- `--notes-json PATH`: add custom file notes and regex notes.
- `--out PATH`: write to a specific output directory.
- `--title TEXT`: override the artifact title.
- `--surrounding-lines N`: number of extra unchanged lines available behind each hunk's expand buttons. Defaults to 12.

Do not use Node or bundlers for this skill. The output should remain plain HTML, CSS, and JavaScript in one file per language, except for approved pinned CDN assets such as `highlight.js` used for syntax highlighting.
