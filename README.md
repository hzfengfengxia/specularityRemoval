##  **Joint Specular Highlight Detection and Removal in Single Images via Unet-Transformer** 



<img src="E:\高光去除-poster扩展工作\fig\1-1.png" alt="1-1" style="zoom:48%;" />

​                                              (a)                                  (b)                                  (c)

Figure: The proposed specular highlight removal and detection results.   (a)  input  specular  highlight  images,  (b)  and  (c)  are highlight removal and detection results, respectively. 

## Requirements

- Python   3.7
- PyTorch 1.7.0
- Cuda 10.0

### Usage

------

SHIQ Dataset, the link is:

 https://drive.google.com/file/d/1RFiNpziz8X5qYPVJPl8Y3nRbfbWVoDCC/view?usp=sharing  

### Training

------

- Modify gpu id, dataset path, and checkpoint path. Adjusting some other parameters if you like.

-  Please run the following code: 

  ```
  python shiq.py --gpus 5\--id UNet --model UNet \
      --optim Adam --lr 1e-4\
      --epochs 80 --batchsize 4 --threads 16
  ```

  

### Testing

------

- Modify test dataset path and result path.

-  Please run the following code: 

  ```
  python shiq.py --gpus 0 --id UNet --model UNet --mode test --epochs 60 --saveimg true
  ```

  