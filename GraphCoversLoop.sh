#!/usr/local/bin/bash
echo $BASH_VERSION

for DEGREE in {3..10}
do
  for NB_COVERS in {3..30..3}
  do
    echo "Running for degree=$DEGREE and nb_covers=$NB_COVERS"
    sed -i '' "7s/.*/  format: PyG-GraphCovers-$DEGREE-$NB_COVERS/" configs/GPS/graphcovers-GPS+RWSE.yaml
    # Run with updated config (not printing results in console but in log file)
    python main.py --cfg configs/GPS/graphcovers-GPS+RWSE.yaml wandb.use False > results/graphcovers-history/logs.log
    # Store results
    DIR="results/graphcovers-history/d${DEGREE}-n${NB_COVERS}/"
    mkdir -p $DIR
    cp -R results/graphcovers-GPS+RWSE/0/* $DIR
    cp results/graphcovers-GPS+RWSE/config.yaml $DIR
  done
done
