python train.py --train-data "./data/meta/train_tf.txt" --test-data "./data/meta/test_tf.txt" --data-base "./data/ShapeNetP2M" --skip-train --checkpoint "./checkpoints/full-model-40-epochs.pth" && echo "40 EPOCH FULL MODEL EVALUATION ON [ALL CATEGORIES] COMPLETED"

python train.py --train-data "./data/meta/train_tf.txt" --test-data "./data/meta/category/test_list_plane.txt" --data-base "./data/ShapeNetP2M" --skip-train --checkpoint "./checkpoints/full-model-40-epochs.pth" && echo "40 EPOCH FULL MODEL EVALUATION ON [PLANE] COMPLETED"

python train.py --train-data "./data/meta/train_tf.txt" --test-data "./data/meta/category/test_list_bench.txt" --data-base "./data/ShapeNetP2M" --skip-train --checkpoint "./checkpoints/full-model-40-epochs.pth" && echo "40 EPOCH FULL MODEL EVALUATION ON [BENCH] COMPLETED"

python train.py --train-data "./data/meta/train_tf.txt" --test-data "./data/meta/category/test_list_cabinet.txt" --data-base "./data/ShapeNetP2M" --skip-train --checkpoint "./checkpoints/full-model-40-epochs.pth" && echo "40 EPOCH FULL MODEL EVALUATION ON [CABINET] COMPLETED"

python train.py --train-data "./data/meta/train_tf.txt" --test-data "./data/meta/category/test_list_car.txt" --data-base "./data/ShapeNetP2M" --skip-train --checkpoint "./checkpoints/full-model-40-epochs.pth" && echo "40 EPOCH FULL MODEL EVALUATION ON [CAR] COMPLETED"

python train.py --train-data "./data/meta/train_tf.txt" --test-data "./data/meta/category/test_list_chair.txt" --data-base "./data/ShapeNetP2M" --skip-train --checkpoint "./checkpoints/full-model-40-epochs.pth" && echo "40 EPOCH FULL MODEL EVALUATION ON [CHAIR] COMPLETED"

python train.py --train-data "./data/meta/train_tf.txt" --test-data "./data/meta/category/test_list_monitor.txt" --data-base "./data/ShapeNetP2M" --skip-train --checkpoint "./checkpoints/full-model-40-epochs.pth" && echo "40 EPOCH FULL MODEL EVALUATION ON [MONITOR] COMPLETED"

python train.py --train-data "./data/meta/train_tf.txt" --test-data "./data/meta/category/test_list_lamp.txt" --data-base "./data/ShapeNetP2M" --skip-train --checkpoint "./checkpoints/full-model-40-epochs.pth" && echo "40 EPOCH FULL MODEL EVALUATION ON [LAMP] COMPLETED"

python train.py --train-data "./data/meta/train_tf.txt" --test-data "./data/meta/category/test_list_speaker.txt" --data-base "./data/ShapeNetP2M" --skip-train --checkpoint "./checkpoints/full-model-40-epochs.pth" && echo "40 EPOCH FULL MODEL EVALUATION ON [SPEAKER] COMPLETED"

python train.py --train-data "./data/meta/train_tf.txt" --test-data "./data/meta/category/test_list_firearm.txt" --data-base "./data/ShapeNetP2M" --skip-train --checkpoint "./checkpoints/full-model-40-epochs.pth" && echo "40 EPOCH FULL MODEL EVALUATION ON [FIREARM] COMPLETED"

python train.py --train-data "./data/meta/train_tf.txt" --test-data "./data/meta/category/test_list_couch.txt" --data-base "./data/ShapeNetP2M" --skip-train --checkpoint "./checkpoints/full-model-40-epochs.pth" && echo "40 EPOCH FULL MODEL EVALUATION ON [COUCH] COMPLETED"

python train.py --train-data "./data/meta/train_tf.txt" --test-data "./data/meta/category/test_list_table.txt" --data-base "./data/ShapeNetP2M" --skip-train --checkpoint "./checkpoints/full-model-40-epochs.pth" && echo "40 EPOCH FULL MODEL EVALUATION ON [TABLE] COMPLETED"

python train.py --train-data "./data/meta/train_tf.txt" --test-data "./data/meta/category/test_list_cellphone.txt" --data-base "./data/ShapeNetP2M" --skip-train --checkpoint "./checkpoints/full-model-40-epochs.pth" && echo "40 EPOCH FULL MODEL EVALUATION ON [CELLPHONE] COMPLETED"

python train.py --train-data "./data/meta/train_tf.txt" --test-data "./data/meta/category/test_list_watercraft.txt" --data-base "./data/ShapeNetP2M" --skip-train --checkpoint "./checkpoints/full-model-40-epochs.pth" && echo "40 EPOCH FULL MODEL EVALUATION ON [WATERCRAFT] COMPLETED"

# TODO: add ablation study model