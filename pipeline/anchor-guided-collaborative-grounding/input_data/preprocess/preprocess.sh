

conda   init
conda   activate   your_env_name

# change the following input and output file names as needed

# Convert RiVEG-VG preprocessed TXT to JSONL
python to_jsonl.py --input_txt RiVEG_VG_stage_preprocessed_file.txt --output_jsonl converted.jsonl

# task: fmnerg or gmner
# Merge CONLL text data into the JSONL file
python merge_conll_text.py \
  --task fmnerg \
  --input_jsonl converted.jsonl \
  --conll_file ancor/data/fmnerg/test.txt \
  --output_file ve_pred_with_text.jsonl

# Merge with entity knowledge
python merge_with_kn.py \
  --input_file ve_pred_with_text.jsonl \
  --jsonl_with_knowledge path_to_jsonl_with_knowledge.jsonl \
  --output_file final_output.jsonl