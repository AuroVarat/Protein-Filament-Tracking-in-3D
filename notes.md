good training video
ch20_URA7_URA8_001_hyperstack_crop_10_ch1_5z
ch20_URA7_URA8_002_hyperstack_crop_48_ch1_5z


(base) auro@apat1511:~/biohack/biohack$ uv run python scripts/train_3d.py
No volumes specified. Discovering from masks...
Discovered 1 TIFF files: tiffs3d/ch20_URA7_URA8_002_hyperstack_crop_57.tif
Loading tiffs3d/ch20_URA7_URA8_002_hyperstack_crop_57.tif...
Loaded 14 explicitly annotated 3D volumes (14 positive + 0 empty masks)

3D Model params: 1,358,961 (cuda)
Class balance pos_weight = 10.0 (raw=1434)

Training for 30 epochs...
  Epoch  1/30  Loss:0.6863  Dice:0.073
  Epoch  5/30  Loss:0.4742  Dice:0.382
  Epoch 10/30  Loss:0.3729  Dice:0.447
  Epoch 15/30  Loss:0.3225  Dice:0.519
  Epoch 20/30  Loss:0.2890  Dice:0.506
  Epoch 25/30  Loss:0.2198  Dice:0.630
  Epoch 30/30  Loss:0.1130  Dice:0.724

Saved trained 3D model to models/filament_unet3d.pt (Final Dice: 0.724)


Loaded 14 explicitly annotated 3D volumes (14 positive + 0 empty masks)

3D Model params (1-ch, Auto-Threshold): 1,358,961 (cuda)
Class balance pos_weight = 10.0 (raw=1603)

Training for 30 epochs...
  Epoch  1/30  Loss:0.7364  Dice:0.059
  Epoch  5/30  Loss:0.5203  Dice:0.373
  Epoch 10/30  Loss:0.3886  Dice:0.434
  Epoch 15/30  Loss:0.3352  Dice:0.475
  Epoch 20/30  Loss:0.3081  Dice:0.510
  Epoch 25/30  Loss:0.2823  Dice:0.528
  Epoch 30/30  Loss:0.2218  Dice:0.584

Saved trained 3D model to models/filament_unet3d_auto.pt (Final Dice: 0.584)