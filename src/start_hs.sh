#!/bin/bash


for lr in 0.000001 0.000003 0.00001 0.00003
do
	python3.8 -u main.py \
	--batch_size 64 \
	--lr $lr \
	--random_seed 123 \
	--n_epochs 20 \
	--data "$1" \
	--tok_type "$2" \
	--device "$3" \
	--hs
done
