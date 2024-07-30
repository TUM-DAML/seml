import toml
from ruamel.yaml import YAML

with open('pyproject.toml') as inp:
    pyproject = toml.load(inp)

deps = (
    pyproject['project']['dependencies']
    + pyproject['project']['optional-dependencies']['dev']
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
