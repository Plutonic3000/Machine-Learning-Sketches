#!/bin/bash
pax=( 'numpy' 'matplotlib' 'pandas' 'scikit-learn' 'scipy' 'jupyter' 'theano' 'keras' 'nltk' 'seaborn' 'flask' 'pyprind' 'wtforms' 'mlxtend' 'PyQt5' 'spyder' )

for i in "${pax[@]}"
do
    sudo pip3 install $i
done
