# --1--
bash run_hpo_glue.sh 0 0.2 sst2 &
bash run_hpo_glue.sh 1 0.4 sst2 &

# --2--
bash run_hpo_glue.sh 0 0.6 sst2 &
bash run_hpo_glue.sh 1 0.8 sst2 &

# --3--
bash run_hpo_glue.sh 0 1.0 sst2 &
bash run_hpo_glue.sh 1 0.2 cola &

# --4--
bash run_hpo_glue.sh 0 0.4 cola &
bash run_hpo_glue.sh 1 0.6 cola &

# --5--
bash run_hpo_glue.sh 0 0.8 cola &
bash run_hpo_glue.sh 1 1.0 cola &
