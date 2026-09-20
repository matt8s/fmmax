# FMMAX maintenance

## Status and responsibilities

As of 2026-09-20, this repository is a community-maintenance fork led by
[@matt8s](https://github.com/matt8s). The original repository is archived.
This does not establish a transfer of the original GitHub repository, PyPI
project, or documentation hosting. Original authors Martin Schubert and Alec
Hammond remain credited; the MIT license and existing notices are preserved.

Please open new issues and pull requests in `matt8s/fmmax`. When continuing
upstream work, link the original issue or PR and summarize its current status.
An open upstream issue is not evidence that a proposed fix is still needed.

## Upstream provenance and discovery

- Original source: <https://github.com/facebookresearch/fmmax>
- Inherited main: `ec1de1899480674364a9825c547db0ba67b5b065`
- Maintenance discussion (locked): <https://github.com/facebookresearch/fmmax/issues/140>
- Branches: <https://github.com/facebookresearch/fmmax/branches/all>
- Tags: <https://github.com/facebookresearch/fmmax/tags>
- Issues: <https://github.com/facebookresearch/fmmax/issues>
- Pull requests: <https://github.com/facebookresearch/fmmax/pulls>
- Fork network: <https://github.com/facebookresearch/fmmax/forks>
- Fork comparison and inventory: [FORKS.md](FORKS.md)
- Existing docs: <https://facebookresearch.github.io/fmmax/>
- Existing package: <https://pypi.org/project/fmmax/>

For local review, fetch upstream branches and pull-request heads into distinct
remote-tracking namespaces. Keep other forks under separate remotes so branch
names and provenance remain unambiguous. Compare changes before selecting them;
fork activity alone does not imply correctness or compatibility.

```sh
git remote add upstream https://github.com/facebookresearch/fmmax.git
git fetch upstream '+refs/heads/*:refs/remotes/upstream/*'
git fetch upstream '+refs/pull/*/head:refs/remotes/upstream-pr/*'
git log --oneline main..upstream-pr/152
```

Skip `remote add` if upstream is already configured. Use explicit branch pushes
for reviewed contributions rather than mirroring the upstream network.

## Initial priorities

1. Establish a reproducible test baseline with the inherited JAX constraint.
2. Review JAX modernization and gradient/eigensolver compatibility before
   changing dependency bounds (upstream PR #152).
3. Triage the crystal magnetic-field example (#120), GPU vector-formulation
   performance (#88), and physical validation cases (#26).
4. Review documentation dependencies, build reproducibility, and hosting
   (#62, #104), then enable this fork's documentation deployment.
5. Review unmerged upstream and fork contributions with attribution and tests.

Issue numbers above refer to `facebookresearch/fmmax`, not this fork.

## Handover and releases

The original owners must approve any upstream transfer and coordinate with the
organization. PyPI roles must be granted separately by current package owners.
Until then, the existing PyPI distribution remains upstream's distribution.

Release publication is gated by the repository variable
`FMMAX_PUBLISH_ENABLED=true`; leave it unset until publishing access, credentials,
versioning, and release tests have been verified. Documentation deployment is
similarly gated by `FMMAX_DOCS_DEPLOY=true` until hosting and site configuration
are ready. Neither gate should be enabled merely to make a workflow green.

Before a release, run the complete test matrix (including optional `jeig`),
review dependency compatibility, build and inspect source/wheel distributions,
verify documentation, and write release notes with contributor credit. Preserve
historical tags; do not retag inherited releases.
