# Local security backport

This directory contains the published `braces` 3.0.3 package with a narrowly
maintained backport for CVE-2026-93687 / GHSA-vfj7-8cjw-p6xm. The package keeps
its original name, authorship, MIT license, and source metadata; the local
prerelease version is `3.0.4-fmmax.0` and is not an upstream release.
The base archive is the npm `braces@3.0.3` artifact with integrity
`sha512-yQbXgO/OSZVD2IsiLlro+7Hf6Q18EJrKSEsdoMzKePKXct3gvD8oLcOQdIzGupr5Fj+EDe8gO/lxc1BzfMpxvA==`.

The depth guards are based on the approach proposed in upstream pull request
micromatch/braces#72 at commit `d0d575e55e74a4e0218e5248fafb79efc3e54ebb`.
This backport additionally normalizes fractional and negative limits and
preserves the published 3.0.3 `stringify(..., {escapeInvalid: true})` behavior.

Remove this directory and its npm override once an upstream patched release is
available and has passed the documentation build and security regressions.
