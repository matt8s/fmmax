/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */
// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

module.exports = async function createConfig() {
  const math = (await import('remark-math')).default;
  const katex = (await import('rehype-katex')).default;

  /** @type {import('@docusaurus/types').Config} */
  const config = {
    title: 'FMMAX Documentation',
    tagline: 'Differentiable Fourier modal simulations with JAX',
    // favicon: 'img/favicon.ico',

    // Set the production url of your site here
    url: 'https://matt8s.github.io/',
    // Set the /<baseUrl>/ pathname under which your site is served
    // For GitHub pages deployment, it is often '/<projectName>/'
    baseUrl: '/fmmax/',

    // GitHub pages deployment config.
    // If you aren't using GitHub pages, you don't need these.
    organizationName: 'matt8s', // Usually your GitHub org/user name.
    projectName: 'fmmax', // Usually your repo name.

    onBrokenLinks: 'throw',
    markdown: {
      // Generated API and tutorial files are standard Markdown rather than MDX.
      format: 'detect',
      hooks: {
        onBrokenMarkdownLinks: 'warn',
      },
    },

    presets: [
      [
        'classic',
        /** @type {import('@docusaurus/preset-classic').Options} */
        ({
          docs: {
            sidebarPath: require.resolve('./sidebars.js'),
            remarkPlugins: [math],
            rehypePlugins: [katex],
            routeBasePath: '/',
          },
          blog: false,
          theme: {},
        }),
      ],
    ],
    stylesheets: [
      {
        href: 'https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/katex.min.css',
        type: 'text/css',
        integrity:
          'sha384-nH0MfJ44wi1dd7w6jinlyBgljjS8EJAh2JBoRad8a3VDw2K69vfaaqm4WnR+gXtA',
        crossorigin: 'anonymous',
      },
    ],
    themeConfig:
      /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
      ({
        image: 'img/docusaurus-social-card.jpg',
        navbar: {
          title: 'FMMAX',
          items: [
            {
              type: 'docSidebar',
              sidebarId: 'docsSidebar',
              position: 'left',
              label: 'Docs',
            },
            {
              type: 'docSidebar',
              sidebarId: 'APISidebar',
              position: 'left',
              label: 'API Reference',
            },
            {
              href: 'https://github.com/matt8s/fmmax',
              label: 'GitHub',
              position: 'right',
            },
          ],
        },
        footer: {
          style: 'dark',
          links: [
            {
              title: 'Project',
              items: [
                {
                  label: 'GitHub',
                  href: 'https://github.com/matt8s/fmmax',
                },
                {
                  label: 'License',
                  href: 'https://github.com/matt8s/fmmax/blob/main/LICENSE',
                },
                {
                  label: 'Original project',
                  href: 'https://github.com/facebookresearch/fmmax',
                },
              ],
            },
          ],
          copyright: `FMMAX contributors. Original project copyright © Meta Platforms, Inc. Built with Docusaurus.`,
        },
      }),
  };

  return config;
};
