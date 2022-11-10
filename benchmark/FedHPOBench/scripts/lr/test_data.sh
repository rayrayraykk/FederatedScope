set -e

cudaid=$1
dataset=$2
alpha=$3

cd ../../../..

out_dir=out_${dataset} 

if [ ! -d $out_dir ];then
  mkdir $out_dir
fi

if [[ $dataset = '10101' ]]; then
    out_channels=2
elif [[ $dataset = '53' ]]; then
    out_channels=4
elif [[ $dataset = '146818' ]]; then
    out_channels=2
elif [[ $dataset = '146821' ]]; then
    out_channels=4
elif [[ $dataset = '146822' ]]; then
    out_channels=7
elif [[ $dataset = '31' ]]; then
    out_channels=2
elif [[ $dataset = '3917' ]]; then
    out_channels=2
else
    out_channels=2
fi

echo "HPO starts..."

sample_rates=(1.0)
lrs=(0.00001)
wds=(0.0)
steps=(1)
batch_sizes=(256)
datasets=(10101 53 146818 146821 146822 31 3917)

for (( sr=0; sr<${#sample_rates[@]}; sr++ ))
do
    for (( l=0; l<${#lrs[@]}; l++ ))
    do
        for (( w=0; w<${#wds[@]}; w++ ))
        do
            for (( s=0; s<${#steps[@]}; s++ ))
            do
                for (( b=0; b<${#batch_sizes[@]}; b++ ))
                do
                    for k in {1..3}
                    do
                        for (( ds=0; ds<${#datasets[@]}; ds++ ))
                        do
                            python federatedscope/main.py --cfg benchmark/FedHPOBench/scripts/lr/openml_lr.yaml device $cudaid data.splitter_args \[\{\'alpha\'\:${alpha}\}\] train.optimizer.lr ${lrs[$l]} train.optimizer.weight_decay ${wds[$w]} train.local_update_steps ${steps[$s]} data.type ${datasets[$ds]}@openml data.batch_size ${batch_sizes[$b]} federate.sample_client_rate ${sample_rates[$sr]} model.out_channels $out_channels seed $k outdir lr_test/${out_dir}_${sample_rates[$sr]} expname lr${lrs[$l]}_wd${wds[$w]}_dropout0_step${steps[$s]}_batch${batch_sizes[$b]}_seed${k}
                        done
                    done
                done
            done
        done
    done
done

echo "HPO ends."
