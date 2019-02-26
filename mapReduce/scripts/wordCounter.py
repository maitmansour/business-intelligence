#!/usr/bin/python
# -*- coding: utf-8 -*-
import mrs
import glob
import re

class wordCounter(mrs.MapReduce):

        def input_data(self, job):
        	# Check if args are used
            if len(self.args) < 2:
                print >> sys.stderr, 'USAGE : wordCounter inputDirectory/ outputDirectory/'
                return None
            inputs = []
            # Save input directory name
            inputDirectory=self.args[0]
            # Get files
            files=glob.glob(inputDirectory+'*.txt', recursive=True)
            print('***** Input Files *****')
            print(files)
            return job.file_data(files)
        def map(self,key, value):
        	# Get words (lowercase) from string
            regex = r'\w+'
            wordsList=re.findall(regex,value.lower())
            newKey=1
            print('***** Input Words *****')
            print(wordsList)            
            for word in wordsList:
                yield (word, newKey)
        def reduce(self,key, values):
        	# Return values sum
            sum_values=sum(values)
            print('***** Sum Values *****')
            print(sum_values)
            yield sum_values


if __name__ == '__main__':
    mrs.main(wordCounter)
