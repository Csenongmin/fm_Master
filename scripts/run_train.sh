
python /mnt/data/soccer_event_no_subtype_refactor/train_only.py \
  --data_pt /mnt/data/soccer_event_no_subtype_refactor/artifacts/dataset.pt \
  --save_dir /mnt/data/soccer_event_no_subtype_refactor/artifacts \
  --d_in 11 --d_model 128 --heads 4 --layers 2 --temp_layers 4 \
  --batch_size 4 --lr 1e-3 --weight_decay 1e-2 --max_steps 200
