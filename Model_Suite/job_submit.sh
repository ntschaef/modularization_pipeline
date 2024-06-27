#!/bin/bash -e

JOBDIR=$(mktemp -d job_XXXX)

cd $JOBDIR

echo Running from $JOBDIR

cp -r ../abstention/ $PWD
cp -r ../data_loaders/ $PWD
cp -r ../keywords/ $PWD
cp -r ../models/ $PWD
cp -r ../predict/ $PWD
cp -r ../training/ $PWD
cp -r ../validate/ $PWD

cp ../run_*.py $PWD
cp ../*args.yml $PWD
cp ../train_model.py $PWD
cp ../new_IE.sh $PWD
sbatch new_IE.sh
