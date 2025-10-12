#for Times in $(seq 1 8) 
#do
Times=1.2
  #for Temp in $(seq 0.5 0.1 1.5)  # temperature:0.5-1.5
  #do
  Temp=7
    python /TempParaphraser/attack/paraphraser.py \
      --eval_model_name hc3 \
      --rewrite_times $Times \
      --temperature $Temp \
      --input_path /data/HC3/test_rnd10k.jsonl \
      --save_path /TempParaphraser/result/llama3.2-1b/n$Times-T$Temp-hc3.json \
      --api http://0.0.0.0:8000/v1
  #done
#done
