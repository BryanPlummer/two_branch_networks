wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P ./data/
unzip ./data/captions_train-val2014.zip -d ./data/coco
rm ./data/captions_train-val2014.zip

wget http://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k.tar.gz -P ./data/
tar zxvf ./data/flickr30k.tar.gz -C ./data/flickr/
rm ./data/flickr30k.tar.gz
