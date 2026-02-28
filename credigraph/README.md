# Maintenance 

To release a new version: 
```bash
# commit to git first, then bump version 
bump2version patch   # for bug fixes
bump2version minor   # for new features  
bump2version major   # for breaking changes
```

To then update the PyPI Package: 

```bash
rm -rf dist/
python -m build
twine upload dist/*
```

To test, see user README.