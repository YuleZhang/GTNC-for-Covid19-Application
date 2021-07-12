# Environment

pip install -r requirements.txt 


You can run start_train_gtnc.py in python3 and the parameters can be changed in Parameters.py.


By default, cpu is used for calculation. GPU can also be used in start_train_gtnc.py. (see it for details) 


Here are the version informations:

Python 3.6.8

numpy 1.16.0

spicy 0.16.0

torch 1.1.0

torchvision 0.3.0

scikit-image 0.15.0

opencv-python 4.1.0.25

nvidia-ml-py3 7.352.0

# Dataset

2020-12-11

Today i find a new dataset from **kaggle** "[COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)",here are some details about this dataset.

There are 219 COVID-19 positive images in this dataset, 1341 normal images and 1345 viral pneumonia images. It will continue to update this database as soon as we have new x-ray images for COVID-19 pneumonia patients.And this dataset is just a expansion of the original one.So it can be easily used into the GTNC or others work that have been down by our group.Here are some CT image examples

![ct-image](https://github.com/SWUQML/GTNC-for-Covid19-Application/blob/master/images/CT_Image.png)

# Details

Here are my learning report about this project:  [Here](http://www.yulezhang.com/2020/12/06/50Study-Report(1130-1206)/)

The report are written in Chinese, including some explanition about this project.In fact, the source of this code are created by Sun Zhengzhi and S.J. Ran, both of them are famous and they have contributed a lot to tensor network machine learning.You can access the original code of this project by click [here](https://github.com/crazybigcat/GTNC), and I have modify some tools of this project.

In my opinion, this project can be easily reused by other projects with modifying a little parameters.And the combination of mutiple parameters can be saved to MD5 file after training, so it's easy for you to retrain them in a short time(just loading the model).

## **Cost function**

When i am reading the code, i try to write the latex according the code as follows:

![img](https://latex.codecogs.com/gif.latex?2log(\sqrt{\sum_{i=1}^{n}%20\sum_{j=1}^{n}\left|T_{ivj}\right|^{2}})-log(N_c)-\frac{2}{N_c}\sum_{m}{(E_{pm}+log(\sum_{pv}{\mid%20I_{mpv}T_{ivj}E_{m,v}}\mid))})

but the latex in [paper](https://arxiv.org/abs/1903.10742) is:

![img](https://latex.codecogs.com/gif.latex?f=-\frac{1}{A}%20\sum_{X%20\in%20\mathcal{A}}%20\log(\frac{\left\langle%20X|\psi\rangle^{2}\right.}{\left\langle%20T^{[n]},%20T^{[\tilde{n}]\rangle}\right\rangle}))

Actually, it's hard for code to correspond  the latex in paper when i try to compare them.But one thing can be denfinitely certained is that **the times of initial loop will be affected by two adjacent loss value**, because we use the function **is_convergence to judge if the loss value are downscaling in a big step**.Apparently, we will stop continuely to train model if the **loss values have no more changes.**

## Result

![ct-image](https://github.com/SWUQML/GTNC-for-Covid19-Application/blob/master/images/result.png)

Here is the result of the classfication, model accuracy up to 94% when cutting bond is 256

