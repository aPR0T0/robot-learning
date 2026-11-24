# site

The node map: a static page that draws every stage, component and pipeline from `graph.json`, and lights up the
files that implement each one. Live at <https://apr0t0.github.io/robot-learning/>.

- `graph.json` is generated: `python scripts/build_site.py` (AST over `src/`, YAML over `configs/`).
- `results.json` holds the verified run numbers it shows.
- Publish with `scripts/publish_site.sh` (copies into the apr0t0.github.io repo).
