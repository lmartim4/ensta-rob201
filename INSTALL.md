## Installation and launch of the code

Works up to python3.12

```bash
git clone https://github.com/emmanuel-battesti/ensta-rob201.git
cd ensta-rob201
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 tp_rob201/main.py
```

## Arcade library dependencies

First, you will obviously have to use the Git tool.

And for the library *Arcade*, you might need to install *libjpeg-dev* and *zlib1g-dev*.

```bash
sudo pacman -Sy libjpeg zlib
```