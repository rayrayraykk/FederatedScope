set -e

cudaid=$1
dataset=$2
lr=$3
seed=$4

# lrs=(0.00001 0.0001 0.001 0.01 0.1 1.0)
# seed=(1 2 3)

cd ../../../..

out_dir=out_${dataset}

if [ ! -d $out_dir ];then
  mkdir $out_dir
fi

echo "HPO starts..."

sample_rates=(1.0)
wds=(0.0 0.001 0.01 0.1)
steps=(1 2 3 4)
batch_sizes=(8 16 32 64 128 256)

for (( sr=0; sr<${#sample_rates[@]}; sr++ ))
do
    for (( w=0; w<${#wds[@]}; w++ ))
    do
        for (( s=0; s<${#steps[@]}; s++ ))
        do
            for (( b=0; b<${#batch_sizes[@]}; b++ ))
            do
                python federatedscope/main.py --cfg benchmark/FedHPOBench/scripts/lr/${dataset}@openml.yaml device $cudaid  train.optimizer.lr ${lr} train.optimizer.weight_decay ${wds[$w]} train.local_update_steps ${steps[$s]} data.type ${dataset}@openml data.batch_size ${batch_sizes[$b]} federate.sample_client_rate ${sample_rates[$sr]} seed ${seed} outdir lr/${out_dir}_${sample_rates[$sr]} expname lr${lr}_wd${wds[$w]}_dropout0_step${steps[$s]}_batch${batch_sizes[$b]}_seed${seed} >/dev/null 2>&1
            done
        done
    done
done

echo "HPO ends."