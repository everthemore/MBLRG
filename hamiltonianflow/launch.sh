#!/bin/bash

output='/home/...'
for i in `seq 0 199`
do
	qsub -v output=$output,L=8,hscale=1,Jscale=4,Uscale=3,seed=$i -N flow-h1-$i controlscript.pbs
	qsub -v output=$output,L=8,hscale=4,Jscale=1,Uscale=1,seed=$i -N flow-h4-$i controlscript.pbs
done
