#!/usr/bin/python
# -*- coding: utf-8 -*-
import mrs
import glob
import re
import sys
import numpy as np
import os

class imcCalculator(mrs.MapReduce):

        def input_data(self, job):
        	# Check if args are used
            if len(self.args) < 2:
                print >> sys.stderr, 'USAGE : imcCalculator inputFile/ outputDirectory/'
                return None
            inputs = []
            # Save input directory name
            inputFile=self.args[0]
            # Get files
            self.reduceFile(inputFile)
            tmp_reduced_files='../data/tmp/'
            files=glob.glob(tmp_reduced_files+'*.csv', recursive=True)
            #print('***** Input Files *****')
            #print(files)
            return job.file_data(files)
        def map(self,key, value):
        	# Get every person values
            values=re.split(',',value)
            if values[0]=='"id"':
                print("Skip first line")
            else:
                newKey=1
                # Skip non adults
                if int(values[1])<18:
                    print("Skip not adult person with age = "+values[1])
                else:
                    # Give key for each sexe
                    if "female" in values[4]:
                        newKey=2
                    del values[4]
                    del values[0]

                    # Calculate IMC
                    imc_value=int(values[1])/(int(values[2])**2)
                    print('***** Input data *****')
                    print(values)            
                    yield (newKey, imc_value)
        def reduce(self,key, values):
        	# Calculate Averages
            lst = list(values)
            average=np.average(lst)
            maximum=max(lst)
            minimum=min(lst)
            # Remove Tmp Files
            self.cleanTmpFiles()
            # Print details
            if key==2:
                result='\nFemale MAX : '+str(maximum)+'\nFemale MIN :'+str(minimum)+'\nFemale AVG :'+str(average)
            else:
                result='\nMale MAX : '+str(maximum)+'\nMale MIN :'+str(minimum)+'\nMale AVG :'+str(average)
            yield result
        def reduceFile(self,filename):
            lines_per_file = 10000
            smallfile = None
            tmp_reduced_files_directory='../data/tmp/'
            print("**** Start Minimizing : "+filename+" ****")
            with open(filename) as bigfile:
                for lineno, line in enumerate(bigfile):
                    if lineno % lines_per_file == 0:
                        if smallfile:
                            smallfile.close()
                        small_filename = tmp_reduced_files_directory+'small_file_{}.csv'.format(lineno + lines_per_file)
                        smallfile = open(small_filename, "w")
                        print("Start Writing : "+small_filename)
                    smallfile.write(line)
                if smallfile:
                    smallfile.close()
        def cleanTmpFiles(self):
            print("**** Clean tmp files ****")
            files=glob.glob('../data/tmp/*.csv', recursive=True)
            for f in files:
                os.remove(f)

if __name__ == '__main__':
    mrs.main(imcCalculator)
