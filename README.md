# Business-intelligence

## Technologies
- mrs.MapReducer
- Python

## Programs
- WordCounter on files
- IMC Avg, Max, Min

## Run Master
``
python <program>.py -I Master -P YYYY --mrs-verbose ../data/sample/sample.txt ../result/
``
## Run Slave
``
python  <program>.py  -I Slave -M X.X.X.X:YYYY --mrs-verbose  
``