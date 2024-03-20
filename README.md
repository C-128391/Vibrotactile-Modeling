#  <p align="center">Research on Low-delay Haptic Texture Display Method Based on Action Information and Texture Image</p>

 <p align="center">Dapeng Chen, Geng Chen, Yi Ding, Qiangqiang Ouyang, Xuhui Hu, Jia Liu, and Aiguo Song</p>
  <p align="center">Nanjing University of Information Science & Technology</p>

## <p align="center">ABSTRACT</p>
In recent years, tool-mediated vibrotactile display of virtual surface texture has become a hot research topic in the field of haptics. When interacting with a virtual texture, it is an effective means to enhance the realism of texture display by incorporating the action information of the user's hand-held tool and the features of the virtual texture into the haptic display process of the texture. To address the problems of weak generalization ability and low interactive realism in texture modeling and rendering, this paper proposes a haptic modeling framework for known textures that varies with action conditions. This framework takes texture images and the user's real-time action information as input. It introduces a self-attention mechanism to assign weights to each feature and combines the previous vibrotactile information to generate corresponding vibrotactile signals. Additionally, we designed a haptic device that integrates a real-time collection of action information and vibrotactile expression capabilities in combination with the 3D Systems Touch device, which together with the haptic modeling framework, forms a texture haptic display system. Based on this, we conducted three user experiments. The results show that our method not only has a certain generalization ability for new textures outside the database, but also can obtain a higher perceptual similarity score. In addition, the delay time of the system we tested is only 29～37 ms, which can bring users a more realistic texture perception experience. 

## <p align="center">THE HAPTIC TEXTURE MODELING FRAMEWORK</p>
In this work, our objective is to establish a multimodal fusion model for predicting tactile signals. This model employs multimodal information (visual images ${x}$, scanning velocity ${v}$, and normal force ${f}$) as inputs to predict the corresponding vibrotactile signals ${a}$. This can be simplified as ${a=g(x, v, f)}$ with ${g}$ representing the prediction model. Inspired by existing works, we designed a haptic texture modeling framework based on action information, the overall structure of the framework is shown below.

![image](https://github.com/C-128391/Vibrotactile-Modeling/blob/main/The%20structure%20of%20haptic%20texture%20rendering%20model.png)

## <p align="center">DETAILS OF IMPLEMENT</p>
This project includes scripts for data preparation, model training, testing, and folders for generated results.
### Dataset
We use the HaTT database as a testing and validation library for the tactile texture modeling framework. The HaTT library contains 100 different texture images in 10 categories, as well as data such as normal force, velocity and vibration acceleration recorded by the experimenter's handheld tool moving on each textured surface in a natural way for 10 seconds. 
For more details, see https://ieeexplore.ieee.org/abstract/document/6775475
### Data Preparation
Start by running `prepair.py` to downsample and normalize the original dataset.
```bash
python prepair.py
```
After dividing the dataset into a training set and test set, use `pre_deal.py` to augment the downsampled data and save it as an Excel file.
```bash
python pre_deal.py
```
### Model Training
Use `train.py` to train the model. Adjust parameters such as epoch and batch_size based on the training environment.
```bash
python train.py
```
After training, the model will be saved in the specified directory.

## Testing
Use `test.py` to validate results on the test set.
```bash
python test.py
```

### Knowledge Distillation Model
The "generate results_teacher model" folder contains the original results generated by the model.
Use `stu-model.py` to train the student model, obtaining a complete Loss curve and the trained model.
```bash
python stu-model.py
```
View the Loss curve with “Loss.png”.
### Get t-SNE image
Use `vision.py` to get t-SNE images
```bash
python vision.py
```
View the original results in the "t-SNE results" folder.

## Real Texture Modeling
Before using the model for real texture modeling, run `image-adjust.py` to adjust the format of images corresponding to real textures, ensuring consistency with the HaTT database's image format.

```bash
python image-adjust.py
```
Then, use the trained model to model textures.

## Result Comparison
Compare t-SNE images obtained using the student model and the teacher model. View the original images in the "t-SNE results" folder.
