# Releasing `workingset` to PyPI

`.github/workflows/publish.yml` builds and publishes on a version tag. No
token is stored anywhere: PyPI trusts the workflow itself (trusted
publishing, OIDC).

## One-time setup (project owner, in a browser)

1. **PyPI.** Log in, open *Your account → Publishing*, and add a pending
   publisher under "Add a new pending publisher":

   | field | value |
   |---|---|
   | PyPI project name | `workingset` |
   | Owner | `T0mSIlver` |
   | Repository name | `working-set` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

   The name `workingset` was free on 2026-09-07. A pending publisher claims
   it on the first successful upload; nobody else can take it in between.

2. **TestPyPI** (optional but recommended before the first real release):
   the same form at https://test.pypi.org with environment name `testpypi`.

3. **GitHub.** *Settings → Environments*: create `pypi` and `testpypi`. On
   `pypi`, add yourself as a required reviewer, so a tag push still needs one
   click before anything reaches the index. Nothing else is needed; the
   workflow requests the OIDC token itself.

## Each release

1. Bump `project.version` in `pyproject.toml`, land it on `main` through the
   normal PR route (CI runs pytest, the golden check and the JS mirror).
2. Dry run to TestPyPI: *Actions → Publish → Run workflow* (target
   `testpypi`), then from a clean machine:

   ```sh
   uvx --index https://test.pypi.org/simple/ --index-strategy unsafe-best-match \
       --from workingset ws models
   ```

3. Tag and push. The tag must equal the version or the build job stops:

   ```sh
   git tag v0.1.0
   git push origin v0.1.0
   ```

   The `build` job runs pytest, builds sdist and wheel, installs the wheel
   into a fresh venv and runs `ws models`; `publish-pypi` waits for the
   environment approval, then uploads.

4. Confirm:

   ```sh
   uvx --from workingset ws predict workingset.toml
   ```

## After the first release

Two places still print the git form of the install command "until the PyPI
release": the explorer's test card (`interactive/src/harness.js`,
`WS_CMD_NOTE`) and the README's install block. Remove the note in a follow-up
PR once `uvx --from workingset ws` resolves.
