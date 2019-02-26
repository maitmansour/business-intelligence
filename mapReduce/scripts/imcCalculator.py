#!/usr/bin/python
# -*- coding: utf-8 -*-
import mrs
import glob
import re
import sys
import numpy as np

class imcCalculator(mrs.MapReduce):
       #if you want to use multiple files
       # def input_data(self, job):
       # 	# Check if args are used
       #     if len(self.args) < 2:
       #         print >> sys.stderr, 'USAGE : imcCalculator inputDirectory/ outputDirectory/'
       #         return None
       #     inputs = []
       #     # Save input directory name
       #     inputDirectory=self.args[0]
       #     # Get files
       #     files=glob.glob(inputDirectory+'*.csv', recursive=True)
       #     print('***** Input Files *****')
       #     print(files)
       #     return job.file_data("../data/socio/socio_100000.csv")
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
                        print(values[4])
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

            # Print details
            if key==2:
                result='\nFemale MAX : '+str(maximum)+'\nFemale MIN :'+str(minimum)+'\nFemale AVG :'+str(average)
            else:
                result='\nMale MAX : '+str(maximum)+'\nMale MIN :'+str(minimum)+'\nMale AVG :'+str(average)
            yield result

if __name__ == '__main__':
    mrs.main(imcCalculator)
