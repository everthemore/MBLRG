#!/bin/bash

#for i in `seq 300 599`
#do
#	qsub -v L=6,hscale=1,Jscale=3,Uscale=1,seed=$i -N flow-W1-$i controlscript.pbs
#	qsub -v L=6,hscale=4,Jscale=1,Uscale=1,seed=$i -N flow-W4-$i controlscript.pbs
#done

for i in `seq 0 299`
do
	qsub -v L=6,hscale=4,Jscale=1,Uscale=1,seed=$i -N flow-W4-$i controlscript.pbs
done

#for i in `seq 0 9`
#do
#	qsub -v L=8,hscale=1,Jscale=3,Uscale=1,seed=$i -N flow-W1-$i controlscript.pbs
#	qsub -v L=8,hscale=4,Jscale=1,Uscale=1,seed=$i -N flow-W4-$i controlscript.pbs
#done
