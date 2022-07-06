#!/bin/bash

#Load the conda module
module load apps/python/conda

source activate pytorch_py3.6
# Run the program <name>.py
# nohup doesn't play well with qsub / qstat,
#  doesn't show up in qstat, doesn't create out.txt
#nohup python FOLLOW_forward_ff_rec_train.py > out.txt

# if you don't redirect output, a <sh file name>.o<qsubid> file contains output
# use -u for unbuffered stdout, i.e. flush after every print
#  else output gets written only after a flush
#  and no flush when a `qdel <qpid>` kill process is sent
python -u fit.py > fit_out56.txt
