# inversion-genesis-mission
Repository for Genesis mission-related code.

| Folder                  | What it contains                                                       | Key tech                                                          |
| ----------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **`lpa/`**| PIC simulations for LPA injectors                                      | FBPIC                                                      |

---

## Quick start

**Prerequisites:** GitHub account with access to this repo, `gh` CLI tool installed and authenticated.

Navigate to the directory you wish to clone this repo on your local machine (these instructions assume macOS)

```bash
# clone (HTTPS method - requires GitHub authentication)
git clone https://github.com/Inversion-Semiconductor/inversion-genesis-mission.git
cd inversion-genesis-mission

# install commit hooks
brew install pre-commit          # if not present
pre-commit install

```

Python baseline: 3.12.x (min 3.11, tests also run on 3.13)

### Recorded Git revision

`pre-commit install` installs the pre-commit checks and the post-commit,
post-checkout, post-merge, and post-rewrite hooks in [.pre-commit-config.yaml](.pre-commit-config.yaml).
**Existing clones must re-run `pre-commit install` after merging these changes**;
updating the configuration does not install new hook stages automatically.
These and FBPIC package builds (including editable pip installs) record `HEAD` in
[lpa/Simulation_FBPIC/inversion_fbpic/git_hash.txt](lpa/Simulation_FBPIC/inversion_fbpic/git_hash.txt).
Build/install the FBPIC subproject, not the repo-root shared-utilities package.
Conda recipes using pip/setuptools use the same hooks; environment creation alone does not.

The file is Git-ignored but bundled in wheels and source distributions. Builds
without Git preserve bundled provenance. For direct, uninstalled source use,
run [tools/record_git_hash.py](tools/record_git_hash.py) before importing.
The rewrite hook covers amend/rebase, but `git reset` (including fetch + reset
workflows) does not run a recording hook. After a reset, run the recorder or
rebuild/reinstall before starting simulations; runtime does not check Git HEAD.

`SerializableConfig` caches the revision at import, keeping running processes
independent of later commits or file updates. Missing, unreadable, or empty files
warn once and yield `git_hash: null` for that process. Refresh before starting a new process;
restart to capture a new revision. The hash identifies committed code, not local edits.

## Development workflow

| Step                        | Command / action                                            |
| --------------------------- | ----------------------------------------------------------- |
| **1. Create ticket branch** | `git switch -c IG-123-short-description`                    |
| **2. Code & commit**        | Hooks run Ruff → Black → clang-format                       |
| **3. Push**                 | `git push -u origin IG-123-short-description`               |
| **4. CI**                   | Branch-name guard + unit tests                              |
| **5. PR**                   | Requires _Code-Owner_ approval; `main` is fast-forward-only |
| **6. Merge**                | PR _Squash & Merge_ →                                       |

Branch names must match `IG-###-short-description`; duplicate ticket IDs are blocked by
`.github/workflows/branch-name.yml`.

## Tooling cheatsheet

- **Poetry** – dependency/packaging (pyproject.toml)
- **Ruff** – linter • **Black** – formatter • **clang-format** – C/C++
- **pytest** – Python tests • **CMake** – C++ builds
- **GitHub Actions** – CI & branch-name guard

## Contributing

1. File/assign a ticket → create branch `IS-###-short-description`.
2. Make sure `pre-commit run --all-files` passes.
3. Open PR; Code-Owner review required.
4. Squash-merge (fast-forward) into main.

## License

© 2026 Inversion Semiconductor.
Internal proprietary research code – redistributing without written permission
is prohibited.

Built with love, vacuum grease, and far too much caffeine. ☕️🔬
