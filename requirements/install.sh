pip install pycocotools
pip install numpy
pip install shapely
pip install spacy
pip install gensim
pip install opencv-python
pip install pygsp
python3 -m spacy download en_core_web_md
DIR="$(cd "$(dirname "$0")" && pwd)"
cd $DIR
wget https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz
gzip -d *.txt.gz 
