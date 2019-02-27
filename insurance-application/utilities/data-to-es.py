#!/usr/bin/python
# -*- coding: utf-8 -*-
##############################################################
# Nom : Mohamed AIT MANSOUR / BELGHARBI Meryem				 #
# Source : M2 ILSEN - Avignon University					 #
# Données créé par : Iain Murray de l’université d’Édimbourg #
##############################################################

# librairies import
import pandas as pd
import sys

# Progress bar function (https://stackoverflow.com/a/15801617/6061900)
def drawProgressBar(percent, barLen = 20):
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()

# read input text and put data inside a data frame
data = pd.read_csv('data/base_prospect.csv', encoding='latin-1')

# Print head
print (data.head())
number_of_rows=len(data)
print("***** Start JSON Creation *****")
json_data=""
current_id=1
for index, row in data.iterrows():
    json_data=json_data+'{"index":{"_index":"insurance_data","_type":"company","_id":'
    json_data=json_data+str(current_id)
    json_data=json_data+'}}\n{"code_cr":"'
    json_data=json_data+str(row['code_cr'])
    json_data=json_data+'","dept":"'
    json_data=json_data+str(row['dept'])
    json_data=json_data+'","effectif":"'
    json_data=json_data+str(row['effectif'])
    json_data=json_data+'","ca_total_FL":"'
    json_data=json_data+str(row['ca_total_FL'])
    json_data=json_data+'","ca_export_FK":"'
    json_data=json_data+str(row['ca_export_FK'])
    json_data=json_data+'","risque":"'
    json_data=json_data+str(row['risque'])
    json_data=json_data+'","endettement":"'
    json_data=json_data+str(row['endettement'])
    json_data=json_data+'","evo_benefice":"'
    json_data=json_data+str(row['evo_benefice'])
    json_data=json_data+'","ratio_benef":"'
    json_data=json_data+str(row['ratio_benef'])
    json_data=json_data+'","evo_effectif":"'
    json_data=json_data+str(row['evo_effectif'])
    json_data=json_data+'","evo_risque":"'
    json_data=json_data+str(row['evo_risque'])
    json_data=json_data+'","age":"'
    json_data=json_data+str(row['age'])
    json_data=json_data+'","type_com":"'
    json_data=json_data+str(row['type_com'])
    json_data=json_data+'","activite":"'
    json_data=json_data+str(row['activite'])
    json_data=json_data+'","actionnaire":"'
    json_data=json_data+str(row['actionnaire'])
    json_data=json_data+'","forme_jur_simpl":"'
    json_data=json_data+str(row['forme_jur_simpl'])
    json_data=json_data+'","chgt_dir":"'
    json_data=json_data+str(row['chgt_dir'])
    json_data=json_data+'","rdv":"'
    json_data=json_data+str(row['rdv'])
    json_data=json_data+'"}\n'
    current_id=current_id+1
    percent=((current_id*100)/number_of_rows)/100
    drawProgressBar(percent)

print("***** End JSON Creation *****")

print("***** Start JSON File Creation *****")
text_file = open("data/base_prospect.json", "w")
text_file.write(json_data)
text_file.close()
print("***** End JSON File Creation *****")
