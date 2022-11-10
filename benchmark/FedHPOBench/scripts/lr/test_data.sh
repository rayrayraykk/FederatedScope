cd ../../../..

python federatedscope/main.py --cfg benchmark/FedHPOBench/scripts/lr/openml_lr.yaml device 1 data.type 10101@openml model.out_channels 2 federate.total_round_num 5
python federatedscope/main.py --cfg benchmark/FedHPOBench/scripts/lr/openml_lr.yaml device 1 data.type 53@openml model.out_channels 4 federate.total_round_num 5
python federatedscope/main.py --cfg benchmark/FedHPOBench/scripts/lr/openml_lr.yaml device 1 data.type 146818@openml model.out_channels 2 federate.total_round_num 5
python federatedscope/main.py --cfg benchmark/FedHPOBench/scripts/lr/openml_lr.yaml device 1 data.type 146821@openml model.out_channels 4 federate.total_round_num 5
python federatedscope/main.py --cfg benchmark/FedHPOBench/scripts/lr/openml_lr.yaml device 1 data.type 146822@openml model.out_channels 7 federate.total_round_num 5
python federatedscope/main.py --cfg benchmark/FedHPOBench/scripts/lr/openml_lr.yaml device 1 data.type 31@openml model.out_channels 2 federate.total_round_num 5
python federatedscope/main.py --cfg benchmark/FedHPOBench/scripts/lr/openml_lr.yaml device 1 data.type 3917@openml model.out_channels 2
