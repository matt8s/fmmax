# FMMAX documentation

This directory contains the Docusaurus site, its hand-written Markdown sources, and the scripts that generate tutorial and API documentation.

## Build and serve the documentation

Run the following commands from the `docs` directory:

```sh
make install
make all
```

`make install` installs the Node and Python dependencies required by the documentation build. It assumes that `python` and `npm` are already available.

`make all` checks the required tools, generates API Markdown from the FMMAX source docstrings, exports the tutorial notebooks to Markdown, and starts the Docusaurus development server. The command remains attached to the server process; use the address printed by Docusaurus to open the site.

While the development server is running, saving a hand-written Markdown file should update the rendered page. Run `make all` again when API docstrings or tutorial notebooks change and their generated Markdown needs to be refreshed.

## Generated content

The documentation build produces two kinds of generated Markdown:

- API pages under `docs/API`, generated from docstrings in `../src/fmmax`;
- tutorial pages under `docs/Tutorials`, generated from notebooks in `../notebooks`.

Edit API documentation in the Python docstrings and tutorial documentation in the notebooks rather than treating the generated Markdown as the source of truth.

The notebook exporter reads the outputs already stored in each notebook. It does not execute notebook cells. Run a tutorial notebook and save its outputs before exporting if those outputs should appear in the documentation.

## Adding a tutorial

1. Add the Jupyter notebook to `../notebooks`.
2. Run the notebook and save any outputs that should be displayed.
3. Run `make all` to generate its Markdown page.
4. Add the generated document ID, without the `.md` suffix, to the Tutorials section of `sidebars.js`.

The order in `sidebars.js` determines the order shown in the Tutorials navigation.

## Cleaning generated files

```sh
make clean
```

This removes ignored files beneath the documentation directory, including generated documentation and installed Node dependencies. Run `make install` before the next build if the dependencies were removed.
