# FMMAX maintenance

## Status and responsibilities

This repository is a community-maintenance fork led by [@matt8s](https://github.com/matt8s). New issues and pull requests for the fork belong in `matt8s/fmmax`.

The original `facebookresearch/fmmax` repository is archived. Maintaining this fork does not transfer ownership of the original repository, the existing PyPI project, or the existing documentation hosting. Original authors Martin Schubert and Alec Hammond remain credited, and the MIT license and existing notices are preserved.

Any upstream ownership transfer must be approved and coordinated by the original owners and organization. Access to publish the existing PyPI project must be granted separately by its current owners. Prospective ownership and publishing arrangements are tracked in the [maintenance transition issue](https://github.com/matt8s/fmmax/issues/1).

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

When continuing upstream or fork work, link the original issue, pull request, and commits, then summarize what remains applicable. An open issue, newer branch, or active fork is a review lead rather than evidence that a change is needed or correct.

For local provenance review, keep upstream branches, pull-request heads, and other forks in distinct remote-tracking namespaces:

```sh
git remote add upstream https://github.com/facebookresearch/fmmax.git
git fetch upstream '+refs/heads/*:refs/remotes/upstream/*'
git fetch upstream '+refs/pull/*/head:refs/remotes/upstream-pr/*'
git log --oneline main..upstream-pr/152
```

Skip `remote add` if upstream is already configured. Publish only explicitly reviewed branches; do not mirror the upstream network.

## How maintenance work is evaluated

Maintenance changes are evaluated on their provenance, demonstrated need, technical correctness, and compatibility rather than their age or source.

- Review the originating discussion and complete diff before adopting work from upstream or another fork.
- Preserve authorship, copyright notices, licenses, and relevant technical history.
- Run the applicable formatting, typing, docstring, test, and documentation checks described in [CONTRIBUTING.md](CONTRIBUTING.md).
- Retain coverage for Python 3.10 / JAX 0.4.38 and Python 3.14 / JAX 0.11.2. Exercise the optional `jeig` backend when eigensolver behavior is affected.
- For numerical changes, test relevant batching, complex dtypes, gradients, conservation laws, field and phase conventions, and convergence or physical regression cases.
- State limitations and unresolved discrepancies directly. Agreement with one implementation or a single passing case is not sufficient evidence for a general numerical claim.

## Publication and documentation deployment

The existing PyPI distribution remains the upstream distribution unless its current owners explicitly grant publishing access. Release publication is gated by the repository variable `FMMAX_PUBLISH_ENABLED=true`. Leave this gate unset until publishing rights, credentials, versioning, and release validation are confirmed.

Documentation deployment is separately gated by `FMMAX_DOCS_DEPLOY=true`. Leave it unset until the fork has an approved hosting target and verified site configuration. Neither gate should be enabled merely to make a workflow pass.

## Release checklist

Before publishing a release:

1. Confirm the required ownership, publishing access, and credentials.
2. Select an appropriate new version and tag. Preserve historical tags and do not retag inherited releases.
3. Run the complete CI matrix, including the supported Python/JAX lanes and the optional `jeig` backend.
4. Review numerical and dependency compatibility for the release.
5. Build and inspect the source and wheel distributions, install them in clean environments, and verify their metadata.
6. Build and review the documentation.
7. Prepare release notes that describe compatibility and limitations and credit all contributors and incorporated work.
8. Enable publication or deployment gates only for the reviewed release.
