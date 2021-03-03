#!/bin/bash

for i in `seq 300 999`
do
	qsub -v L=6,hscale=1,Jscale=4,Uscale=3,seed=$i -N flow-h1-$i controlscript.pbs
	qsub -v L=6,hscale=4,Jscale=1,Uscale=1,seed=$i -N flow-h4-$i controlscript.pbs
done

