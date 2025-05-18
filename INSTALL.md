# Arcade library dependencies

First, you will obviously have to use the Git tool.

And for the library *Arcade*, you might need to install *libjpeg-dev* and *zlib1g-dev*.

For arch linux
```bash
sudo pacman -Sy libjpeg zlib
```

## Installation and launch of the code

Working and tested on python3.10

```bash
git clone https://github.com/lmartim4/ensta-rob201
cd ensta-rob201
python3.10 -m venv .env
source .env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python tp_rob201/main.py
```