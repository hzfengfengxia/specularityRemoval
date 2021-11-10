##  **Joint Specular Highlight Detection and Removal in Single Images via Unet-Transformer** 

Specular highlight detection and removal is a fundamental problem in computer vision and image processing. In this paper, we present an efficient end-to-end deep learning model for simultaneously and automatically detecting and removing specular highlights from a single image. In particular, we first utilize an encoder-decoder network to detect specular highlights, then propose a novel Unet-Transformer network for highlight removal, where we append transformer modules instead of feature maps in the Unet architecture. We also introduce a highlight detection module as a mask to guide the removal task. Thus, these two networks can be jointly trained in an effective manner.
Thanks to the hierarchical and global properties of the transformer mechanism, our framework is able to establish relationships between continuous self-attention layers, making it possible to directly model the mapping relation between the diffuse area and the specular highlight area, and reduce indeterminacy within areas containing strong specular highlight reflection. Experiments on public benchmark and real-world images demonstrate that our approach outperforms state-of-the-art methods for both highlight detection and removal tasks.

## Requirements

- Python   3.7
- PyTorch 1.7.0
- Cuda 10.0

### Usage

------

The link of SHIQ dataset used in the code:

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

  
