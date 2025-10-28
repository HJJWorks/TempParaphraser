#for Times in $(seq 1 8) 
#do
Times=7
  #for Temp in $(seq 0.5 0.1 1.5)  # temperature:0.5-1.5
  #do
  Temp=1.2
    python attack/paraphraser.py \
      --eval_model_name hc3 \
      --rewrite_times $Times \
      --temperature $Temp \
      --input_path test_data/test_rnd10k.jsonl \
      --save_path test_data/n$Times-T$Temp-hc3.json \
      --api http://0.0.0.0:10001/v1
  #done
#done
