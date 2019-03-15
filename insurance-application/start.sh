#!/bin/bash

printf "START PSCRIPTS \n"

echo "\nSCRIPT 1/ : SUPERVISED ALL DATA\n"
python3 scripts/supervised/classifiers-all-numeric.py

echo "\nSCRIPT 2/ : SUPERVISED NUMERIC DATA\n"
python3 scripts/supervised/classifiers-numeric-data.py

echo "\nSCRIPT 3/ : SUPERVISED CATEGORIAL DATA\n"
python3 scripts/supervised/classifiers-categorial-data.py

echo "\nSCRIPT 4/ : TESTING PCA\n"
python3 scripts/testing/analyse-acp.py

echo "\nSCRIPT 5/ : TESTING DECISION TREE\n"
python3 scripts/testing/decision-tree.py

echo "\nSCRIPT 6/ : TESTING LINEAR REGRESSION\n"
python3 scripts/testing/linear-regression.py

echo "\n************ SUPERVISED DONE, PLEASE CHECK LOGS FILES AND PLOTS **************\n"

echo "\nSCRIPT 7/ : UNSUPERVISED EQUILIBRATED DATA KMEANS\n"
python3 scripts/unsupervised/kmeans-equilibrated-data.py

echo "\nSCRIPT 8/ : UNSUPERVISED UNEQUILIBRATED DATA KMEANS\n"
python3 scripts/unsupervised/kmeans-unequilibrated-data.py
