python3 -m venv ve # This only works with Python 3.8 and not 3.9, idk why
source ve/bin/activate
python -m pip install numpy # I don't know why this needs to be installed separately, but it does
python -m pip install -r mini_requirements.txt
python -m spacy download en_core_web_sm
jupyter notebook
