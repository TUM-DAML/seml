import toml

with open('scripts/.pre-commit-config.yaml.template') as inp:
    template = inp.read()

with open('pyproject.toml') as inp:
    pyproject = toml.load(inp)

deps = (
    pyproject['project']['dependencies']
    + pyproject['project']['optional-dependencies']['dev']
)
template = template.format(dependencies=f'[{",".join(deps)}]')

with open('.pre-commit-config.yaml', 'w') as out:
    out.write(template)
