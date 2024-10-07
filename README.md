# HTR2HPC

[![PyPI - Version](https://img.shields.io/pypi/v/htr2hpc.svg)](https://pypi.org/project/htr2hpc)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/htr2hpc.svg)](https://pypi.org/project/htr2hpc)

This repo contains experimental code for customizing eScriptorium.


-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

Install from github:

```console
pip install git+https://github.com/Princeton-CDH/htr2hpc.git@main#egg=htr2hpc
```

### CAS authentication

`pucas` is a dependency of this package and will be automatically installed when you install this package.

1. Import htr2hpc settings into local settings. It must be imported *after* escriptorium settings.
This will add the necessary CAS applications to `INSTALLED_APPS` and sets `ROOT_URLCONF` to use the urls provided with this application.  Add configurations for CAS server url and PUCAS LDAP settings.

```python
from escriptorium.settings import *
from htr2hpc.settings import *

# CAS login configuration
CAS_SERVER_URL = "https://example.com/cas/"

PUCAS_LDAP.update(
    {
        "SERVERS": [
            "ldap2.example.com",
        ],
        "SEARCH_BASE": "",
        "SEARCH_FILTER": "(uid=%(user)s)",
        # other ldap attributes as needed
    }
)
```


## License

`htr2hpc` is distributed under the terms of the Apache 2 license.
