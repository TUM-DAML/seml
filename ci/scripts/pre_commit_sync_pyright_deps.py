#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "toml",
#     "ruamel-yaml",
# ]
# ///
"""
This script reads the pyproject.toml and updates the pyright dependencies in
our pre-commit hook with the current dependencies from the pyproject.toml.
This script itself will be called from a pre-commit hook everytime the pyproject.toml
gets edited.
"""

import toml

# We use ruamel.yaml instead of pyyaml since it preseveres comments and structure.
from ruamel.yaml import YAML

with open('pyproject.toml') as inp:
    pyproject = toml.load(inp)

deps = (
    pyproject['project']['dependencies'] + pyproject['tool']['uv']['dev-dependencies']
)

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)
with open('.pre-commit-config.yaml') as inp:
    template = yaml.load(inp)

# Replacing the dependencies for pyright
for repo in template['repos']:
    if 'pyright' in repo['repo']:
        repo['hooks'][0]['additional_dependencies'] = deps
    else:
        continue

with open('.pre-commit-config.yaml', 'w') as out:
    yaml.dump(template, out)
