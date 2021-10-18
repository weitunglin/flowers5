# Flowers5
This project using the dataset [flowers5](https://www.kaggle.com/alxmamaev/flowers-recognition).

## Result
| Model | Acc |
| ----- | --- |
| Resnet34 | 90.39% |
| Densenet121 | 94.21% | 
| Resnet34 + Densenet121 (freeze backbone) | 94.32% |
| Resnet34 + Densenet121 (finetune) | 93.40% |
| 4 x Densenet121 (freeze backbone) | 95.14% |

## Reference
* https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/5
* https://discuss.pytorch.org/t/custom-ensemble-approach/52024/3
