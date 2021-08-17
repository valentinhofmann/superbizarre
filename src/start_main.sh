#!/bin/bash

for random_seed in {0..19}
do
	python3 -u main.py \
	--batch_size 64 \
	--lr 0.000003 \
	--random_seed $random_seed \
	--n_epochs 20 \
	--data "$1" \
	--tok_type "$2" \
	--device "$3"
done
