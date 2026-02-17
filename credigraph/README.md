# Maintenance 

To release a new version: 
```bash
bump2version patch   # for bug fixes
bump2version minor   # for new features  
bump2version major   # for breaking changes
```

To then update the PyPI Package: 

```bash
python -m build
twine upload dist/*
# and push to git
```