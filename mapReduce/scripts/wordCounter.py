#!/usr/bin/python
# -*- coding: utf-8 -*-
import mrs


class wordCounter(mrs.MapReduce):
	
        def input_data(self, job):
            if len(self.args) < 2:
                print >> sys.stderr, 'USAGE : wordCounter inputDirectory outputDirectory'
                return None
            inputs = []
            f = open(self.args[0])
            for line in f:
                inputs.append(line[:-1])
            return job.file_data(inputs)


if __name__ == '__main__':
    mrs.main(wordCounter)
