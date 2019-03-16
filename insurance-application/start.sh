#!/bin/bash

printf "START PSCRIPTS \n"

echo "\nSCRIPT 1/ : SUPERVISED ALL DATA\n"
python scripts/supervised/classifiers-all-numeric.py

echo "\nSCRIPT 2/ : SUPERVISED NUMERIC DATA\n"
python scripts/supervised/classifiers-numeric-data.py

echo "\nSCRIPT 3/ : SUPERVISED CATEGORIAL DATA\n"
python scripts/supervised/classifiers-categorial-data.py

echo "\nSCRIPT 4/ : TESTING PCA\n"
python scripts/testing/analyse-acp.py

echo "\nSCRIPT 5/ : TESTING DECISION TREE\n"
python scripts/testing/decision-tree.py

echo "\nSCRIPT 6/ : TESTING LINEAR REGRESSION\n"
python scripts/testing/linear-regression.py

echo "\n************ SUPERVISED DONE, PLEASE CHECK LOGS FILES AND PLOTS **************\n"

echo "\nSCRIPT 7/ : UNSUPERVISED EQUILIBRATED DATA KMEANS\n"
python scripts/unsupervised/kmeans-equilibrated-data.py

echo "\nSCRIPT 8/ : UNSUPERVISED UNEQUILIBRATED DATA KMEANS\n"
echo "Skipped, too long"
#python scripts/unsupervised/kmeans-unequilibrated-data.py

echo "\nSCRIPT 9/ : UNSUPERVISED HAC CLASSIFICATION EQUILIBRATED DATA\n"
python scripts/unsupervised/hierarchical-ascending-classification-equilibrated-data.py

echo "\nSCRIPT 10/ : UNSUPERVISED HAC CLASSIFICATION UNEQUILIBRATED DATA\n"
echo "Skipped, too long"
#python scripts/unsupervised/hierarchical-ascending-classification-equilibrated-data.py
