# Vendored documentation libraries

Third-party JavaScript used by the interactive data-model diagram on the [inventory tutorial](../tutorial/inventory.qmd). The files are unmodified releases, checked in rather than fetched from a CDN so that a documentation build works offline and always renders with the versions it was tested against.

`docs/filters/render-data-model.lua` declares them as Quarto HTML dependencies; nothing else in DASCore uses them.

| File | Library | Version | License |
| --- | --- | --- | --- |
| `cytoscape/cytoscape.min.js` | [Cytoscape.js](https://js.cytoscape.org) | 3.30.4 | MIT, in the file header |
| `elk/elk.bundled.js` | [elkjs](https://github.com/kieler/elkjs) | 0.10.0 | EPL-2.0, in `elk/LICENSE-elk` |
| `elk/cytoscape-elk.min.js` | [cytoscape.js-elk](https://github.com/cytoscape/cytoscape.js-elk) | 2.2.0 | MIT, in `elk/LICENSE-cytoscape-elk` |

The license files are deliberately extension-less: quarto renders every `.md` under `docs/` as a page of the site, so `LICENSE.md` would publish the Eclipse Public License at `dascore.org/vendor/elk/LICENSE.html`, inside DASCore's own theme and navigation. Anything added here that is not meant to become a site page needs the same treatment. (`README.md` is exempt — quarto does not render files by that name.)

To upgrade one, replace the file with the new release, update the version in the table and in the `add_html_dependency` call in the filter, and rebuild the docs to confirm the diagram still lays out.
