#cudaid=$1
#dataset=$2
#lr=$3
#seed=$4

# lrs=(0.00001 0.0001 0.001 0.01 0.1 1.0)
# seed=(1 2 3)

nohup bash run.sh 1 31 0.00001 1 &
nohup bash run.sh 2 31 0.00001 2 &
nohup bash run.sh 3 31 0.00001 3 &

nohup bash run.sh 4 31 0.0001 1 &
nohup bash run.sh 5 31 0.0001 2 &
nohup bash run.sh 6 31 0.0001 3 &

nohup bash run.sh 7 31 0.001 1 &
nohup bash run.sh 1 31 0.001 2 &
nohup bash run.sh 2 31 0.001 3 &



nohup bash run.sh 1 31 0.01 1 &
nohup bash run.sh 2 31 0.01 2 &
nohup bash run.sh 3 31 0.01 3 &

nohup bash run.sh 4 31 0.1 1 &
nohup bash run.sh 5 31 0.1 2 &
nohup bash run.sh 6 31 0.1 3 &

nohup bash run.sh 7 31 1.0 1 &
nohup bash run.sh 1 31 1.0 2 &
nohup bash run.sh 2 31 1.0 3 &




nohup bash run.sh 6 10101 0.00001 1 &
nohup bash run.sh 6 10101 0.00001 2 &
nohup bash run.sh 6 10101 0.00001 3 &

nohup bash run.sh 7 10101 0.0001 1 &
nohup bash run.sh 7 10101 0.0001 2 &
nohup bash run.sh 7 10101 0.0001 3 &

nohup bash run.sh 6 10101 0.001 1 &
nohup bash run.sh 6 10101 0.001 2 &
nohup bash run.sh 6 10101 0.001 3 &

nohup bash run.sh 7 10101 0.01 1 &
nohup bash run.sh 7 10101 0.01 2 &
nohup bash run.sh 7 10101 0.01 3 &

nohup bash run.sh 0 10101 0.1 1 &
nohup bash run.sh 1 10101 0.1 2 &
nohup bash run.sh 2 10101 0.1 3 &

nohup bash run.sh 3 10101 1.0 1 &
nohup bash run.sh 4 10101 1.0 2 &
nohup bash run.sh 5 10101 1.0 3 &