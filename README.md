# WSSNet_pytorch_implement
This repository is pytorch's implement of paper: WSSNet: Aortic wall shear stress estimation using deep learning on 4D Flow MRI


Official TensorFolw implementation are here: https://github.com/EdwardFerdian/WSSNet . You can download datasets here and place them in the same level directory as the master like :

```
├── dataset
|   ├── train
|   └── val
|   └── test
├── master
```

In this implementation, it not only includes CNN, but also the implementation of Swin Transformer.

I tried, but the effect was not particularly good.

Unlike the official implementation, I did not include a regular loss term. (Just SSIM + MAE)

Here are some results of SwinT:

| Case                  | MAE  | rel (%) | Pearson |
| --------------------- | ---- | ------- | ------- |
| **Val #1 (normal)**   | 0.59 | 11.67   | 0.57    |
| **Val #2 (normal)**   | 0.42 | 11.67   | 0.71    |
| **Val #3 (normal)**   | 0.48 | 15.16   | 0.64    |
| **Test #1  (normal)** | 0.74 | 13.49   | 0.57    |
| **Test #2  (LVH)**    | 1.07 | 16.54   | 0.70    |
| **Test #3  (normal)** | 0.58 | 12.01   | 0.64    |
| **Overall**           | 0.65 | 13.42   | 0.63    |

Here are some visualizations:

![image-20230419175032294](./assets/image-20230419175032294.png)



In the visualization file, I only modified the file separately: plot_ tawss_ osi_ flatmap.py. Therefore, the work of modifying other visualization files is entrusted to later parties.



You can train by:

```
python main.py
```

you can test some case by:

```
python test.py
```


