# Anacoda Ubuntu PyTorch 環境 #
環境安裝步驟:  
1.顯卡驅動 nvidia-driver（也叫做 cuda driver）：英伟达GPU驱动，命令：nvidia-smi  
2.Anaconda：（或者用阉割版的miniconda）用于创建虚拟环境  
3.python(使用 conda create -n 建立虛擬環境名稱 python=3.8指定版本)  
4.pytorch  
已經使用 pytorch 就不要另外再安裝 cunn  
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102  
5.pip install opencv-python==4.5.3.56	(太新會無法支援)  
6.pip install jupyter notebook(最後放棄了也可以不用裝，Debug能力沒有 VSCode好，改用VSCode 遠端 Debug，但是VSCode 已棄用Python3.6，至少3.7，改使用3.8)  
7.開始訓練  
跑YOLO code需要的套件包  
pip3 install -r requirements.txt  
<br>
<br>
要看YOLO支援的python版本到多少，這關係到全部的安裝  
大多都會寫在 requirements.txt 裡面就可以看出大概使用的版本  
torch == 1.6 => 最低Python 3.6 ~ 3.8  
先看 torch到多少，得到 Python版本後，查詢 PyTorch 支援的 Python版本可用到哪個版本和CUDA的版本  
決定好 CUDA版本(最後選擇 CUDA 10.2，Python是3.6)，採用 pip安裝  
此外多數的YOLO project都會使用到 OpenCV來處理影像，所以這個版本就要看 Python決定使用的版本是多少，也要搭配  
不過OpenCV可以先在隔離的環境下安裝好，這樣再跑 requirements.txt 比較不會出錯  
最後編譯YOLO code的 requirements.txt 會在另外指定要安裝的 torch版本和 numpy等等版本
先不改，先能夠編譯成功在說。  
<br>
<br>
<br>
<br>

# 檔案使用說明 #
### data	資料集或是數據集 70%~80%	(訓練用資料都放在 data下的 標籤過得圖檔資料夾)
	[http://10.1.21.196:3000/YOLO_Developer/YOLOv4-Tiny_Data/data/]  
	即為標籤圖檔(標註資料)時，標出來需要偵測的物件們(name class)  
	其中還可以分出驗證集 20%~30%，驗證用的圖檔，可以在透過標籤工具後另使用不同文字檔分開，
	跑 lableImgML2txt.py 將obj.img 取出全部檔案數量後，取20%、80%的比例(Sample img 的話就可以不用執行，有新的訓練集照片才要重跑)  
	(trainval_files, test_files = train_test_split(image_ids, test_size=0.2, random_state=55) 表示將 20% 的資料分配給驗證集)  
	即分成 train.txt 和 valid.txt  
	obj.names	標籤類別名稱(自己生成，可從 lableImg 處理完的 classes.txt另存)  
	obj.data	定義訓練集圖檔路徑和生成文字路徑，最好使用絕對路徑來避免誤會(自己生成)  
	
	統一用路徑為(原本COCO Set的架構)改成自備好 data/img、data/obj.data、data/obj.names後由程式處理(跑 lableImgML2txt.py)  
	data/img	訓練用圖檔和標籤完成的圖檔資料夾(會有圖檔和ML格式json)	若是壹傳圖檔格式，則要透過 ConvertEtroIMG2SingleID.py   先統一圖檔ID(.jpg)和.json的命名
	data/obj.data	定義訓練集圖檔路徑和生成文字路徑
	data/obj.names	標籤類別名稱
	data/train.txt	訓練集文字檔，改成 Tianxiaomo/pytorch-YOLOv4 PyTorch版本(跑 lableImgML2txt.py)
	data/valid.txt	驗證集文字檔，改成 Tianxiaomo/pytorch-YOLOv4 PyTorch版本(跑 lableImgML2txt.py)  
<br>
<br>

### weight	權重檔	(另外新增 weight資料夾，分類比較清楚，全都放權重檔)	
### weight資料夾[http://10.1.21.196:3000/YOLO_Developer/YOLOv4-Tiny_Data/src/branch/master/weight/]  
	訓練完成的權重檔，之後測試(偵測)各圖檔需要使用的engine  
	通常 yolo裡的 .weights(例如: yolov4.weights) 僅供測試darknet用，後續自定義訓練用不到。  
<br>
<br>

### .pth	pytorch版本的權重檔
	和 weights作用一樣，也可從 weights做轉換變成 pytorch版本  
	
	xxx.conv.數字	自定義預訓練權重檔，也叫做[預訓模型]，自定義訓練使用的  
	從官網  
	https://github.com/AlexeyAB/  Darknet#how-to-train-tiny-yolo-to-detect-your-custom-objects
	How to train (to detect your custom objects)  
	下載自定義預訓練權重檔  
	cfg/yolov4-custom.cfg	yolov4.conv.137(yolov4的)  
	cfg/yolov4-tiny-custom.cfg	yolov4.conv.29(yolov4-tiny的)  
	如果想換成Yolov4或其它系列模型進行訓練，請要修改對應的預訓練檔及修改config檔案裡的 filters  
	So if classes=1 then should be filters=18. If classes=2 then write filters=21. (Do not write in the cfg-file: filters=(classes + 5)x3)  
<br>
<br>
<br>
<br>

# cfg.py 參數設定 # (訓練的時候常會修改到的參數，從 train.py 裡拉到cfg.py處理)  
Cfg.classes_path = 'data/obj.names'  
Cfg.model_path = 'weight/yolov4-tiny.pth'  
Cfg.Unfreeze_batch_size = 2  
Cfg.train_label = os.path.join(_BASE_DIR, 'data', 'train.txt')  
Cfg.val_label   = os.path.join(_BASE_DIR, 'data' ,'valid.txt')  
<br>
<br>
<br>
<br>

# predict.py、train.py、lableImgML2txt.py、ConvertEtroIMG2SingleID.py 帶入的參數 #  
使用權重檔來測試圖片  
Cfg.predict_model_path = 'checkpoints/last_epoch_weights.pth' 調整要使用的權重 last_epoch_weights.pth 或是 best_epoch_weights.pth  
代入 測試圖檔路徑"./data/test1.jpg"  
=> python predict.py "./data/test1.jpg"  
<br>
<br>
開始訓練自定義模型  
不用參數，有調整從 cfg.py 讀出設定的參數  
=> python train.py  
<br>
<br>
將標籤好的圖檔，生成訓練集和驗證集(80%、20%的比例)  
自動跑完生成檔案 /data/train.txt 和 /data/valid.txt  
使用一致，帶入4個參數 -dir", "/data/img", "-clasessname", "/data/obj.names  
=> python lableImgML2txt.py -dir /data/img -clasessname /data/obj.names  
<br>
<br>
將 Etro產品的圖檔命名規則改至單一檔案上，用檔名來當作唯一值，才能夠提供給圖檔 index  
"壹傳格式的圖檔資料夾(年/月/日/車格名稱/車格名稱_時間_第幾張.jpg)", "要存儲的路徑資料夾", "在轉換完成之後是否刪除來源圖檔"  
=> python ConvertEtroIMG2SingleID.py "C:\\Users\\Administrator\\Desktop\\IMG\\SL100\\停管\\錦洲街-民權東路\\2024\\03\\01\\1正常(分類)" "C:\\Users\\Administrator\\Desktop\\IMG\\SL100\\停管\\錦洲街-民權東路\\1正常(編號)" "1"  
<br>
<br>
<br>
<br>

# 以下是原作者的原文  
## YOLOV4-Tiny：You Only Look Once-Tiny目标检测模型在Pytorch当中的实现
---

## 目录
1. [仓库更新 Top News](#仓库更新)
2. [相关仓库 Related code](#相关仓库)
3. [性能情况 Performance](#性能情况)
4. [所需环境 Environment](#所需环境)
5. [文件下载 Download](#文件下载)
6. [训练步骤 How2train](#训练步骤)
7. [预测步骤 How2predict](#预测步骤)
8. [评估步骤 How2eval](#评估步骤)
9. [参考资料 Reference](#Reference)

## Top News
**`2022-04`**:**支持多GPU训练，新增各个种类目标数量计算，新增heatmap。**  

**`2022-03`**:**进行了大幅度的更新，修改了loss组成，使得分类、目标、回归loss的比例合适、支持step、cos学习率下降法、支持adam、sgd优化器选择、支持学习率根据batch_size自适应调整、新增图片裁剪。**
BiliBili视频中的原仓库地址为：https://github.com/bubbliiiing/yolov4-tiny-pytorch/tree/bilibili

**`2021-10`**:**进行了大幅度的更新，增加了大量注释、增加了大量可调整参数、对代码的组成模块进行修改、增加fps、视频预测、批量预测等功能。**    

## 相关仓库
| 模型 | 路径 |
| :----- | :----- |
YoloV3 | https://github.com/bubbliiiing/yolo3-pytorch  
Efficientnet-Yolo3 | https://github.com/bubbliiiing/efficientnet-yolo3-pytorch  
YoloV4 | https://github.com/bubbliiiing/yolov4-pytorch
YoloV4-tiny | https://github.com/bubbliiiing/yolov4-tiny-pytorch
Mobilenet-Yolov4 | https://github.com/bubbliiiing/mobilenet-yolov4-pytorch
YoloV5-V5.0 | https://github.com/bubbliiiing/yolov5-pytorch
YoloV5-V6.1 | https://github.com/bubbliiiing/yolov5-v6.1-pytorch
YoloX | https://github.com/bubbliiiing/yolox-pytorch
YoloV7 | https://github.com/bubbliiiing/yolov7-pytorch
YoloV7-tiny | https://github.com/bubbliiiing/yolov7-tiny-pytorch

## 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| VOC07+12+COCO | [yolov4_tiny_weights_voc.pth](https://github.com/bubbliiiing/yolov4-tiny-pytorch/releases/download/v1.0/yolov4_tiny_weights_voc.pth) | VOC-Test07 | 416x416 | - | 77.8
| VOC07+12+COCO | [yolov4_tiny_weights_voc_SE.pth](https://github.com/bubbliiiing/yolov4-tiny-pytorch/releases/download/v1.0/yolov4_tiny_weights_voc_SE.pth) | VOC-Test07 | 416x416 | - | 78.4
| VOC07+12+COCO | [yolov4_tiny_weights_voc_CBAM.pth](https://github.com/bubbliiiing/yolov4-tiny-pytorch/releases/download/v1.0/yolov4_tiny_weights_voc_CBAM.pth) | VOC-Test07 | 416x416 | - | 78.6
| VOC07+12+COCO | [yolov4_tiny_weights_voc_ECA.pth](https://github.com/bubbliiiing/yolov4-tiny-pytorch/releases/download/v1.0/yolov4_tiny_weights_voc_ECA.pth) | VOC-Test07 | 416x416 | - | 77.6
| COCO-Train2017 | [yolov4_tiny_weights_coco.pth](https://github.com/bubbliiiing/yolov4-tiny-pytorch/releases/download/v1.0/yolov4_tiny_weights_coco.pth) | COCO-Val2017 | 416x416 | 21.5 | 41.0

## 所需环境
torch==1.2.0

## 文件下载
训练所需的各类权值均可在百度网盘中下载。    
链接: https://pan.baidu.com/s/1ABR6lOd0_cs5_2DORrMSRw      
提取码: iauv    

VOC数据集下载地址如下，里面已经包括了训练集、测试集、验证集（与测试集一样），无需再次划分：  
链接: https://pan.baidu.com/s/19Mw2u_df_nBzsC2lg20fQA    
提取码: j5ge  

## 训练步骤
### a、训练VOC07+12数据集
1. 数据集的准备   
**本文使用VOC格式进行训练，训练前需要下载好VOC07+12的数据集，解压后放在根目录**  

2. 数据集的处理   
修改voc_annotation.py里面的annotation_mode=2，运行voc_annotation.py生成根目录下的2007_train.txt和2007_val.txt。   

3. 开始网络训练   
train.py的默认参数用于训练VOC数据集，直接运行train.py即可开始训练。   

4. 训练结果预测   
训练结果预测需要用到两个文件，分别是yolo.py和predict.py。我们首先需要去yolo.py里面修改model_path以及classes_path，这两个参数必须要修改。   
**model_path指向训练好的权值文件，在logs文件夹里。   
classes_path指向检测类别所对应的txt。**   
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。   

### b、训练自己的数据集
1. 数据集的准备  
**本文使用VOC格式进行训练，训练前需要自己制作好数据集，**    
训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。   
训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。   

2. 数据集的处理  
在完成数据集的摆放之后，我们需要利用voc_annotation.py获得训练用的2007_train.txt和2007_val.txt。   
修改voc_annotation.py里面的参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt。   
训练自己的数据集时，可以自己建立一个cls_classes.txt，里面写自己所需要区分的类别。   
model_data/cls_classes.txt文件内容为：      
```python
cat
dog
...
```
修改voc_annotation.py中的classes_path，使其对应cls_classes.txt，并运行voc_annotation.py。  

3. 开始网络训练  
**训练的参数较多，均在train.py中，大家可以在下载库后仔细看注释，其中最重要的部分依然是train.py里的classes_path。**  
**classes_path用于指向检测类别所对应的txt，这个txt和voc_annotation.py里面的txt一样！训练自己的数据集必须要修改！**  
修改完classes_path后就可以运行train.py开始训练了，在训练多个epoch后，权值会生成在logs文件夹中。  

4. 训练结果预测  
训练结果预测需要用到两个文件，分别是yolo.py和predict.py。在yolo.py里面修改model_path以及classes_path。  
**model_path指向训练好的权值文件，在logs文件夹里。  
classes_path指向检测类别所对应的txt。**  
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。  

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载yolo_weights.pth，放入model_data，运行predict.py，输入  
```python
img/street.jpg
```
2. 在predict.py里面进行设置可以进行fps测试和video视频检测。  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"        : 'model_data/yolov4_tiny_weights_coco.pth',
    "classes_path"      : 'model_data/coco_classes.txt',
    #---------------------------------------------------------------------#
    #   anchors_path代表先验框对应的txt文件，一般不修改。
    #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
    #---------------------------------------------------------------------#
    "anchors_path"      : 'model_data/yolo_anchors.txt',
    "anchors_mask"      : [[3,4,5], [1,2,3]],
    #-------------------------------#
    #   所使用的注意力机制的类型
    #   phi = 0为不使用注意力机制
    #   phi = 1为SE
    #   phi = 2为CBAM
    #   phi = 3为ECA
    #-------------------------------#
    "phi"               : 0,  
    #---------------------------------------------------------------------#
    #   输入图片的大小，必须为32的倍数。
    #---------------------------------------------------------------------#
    "input_shape"       : [416, 416],
    #---------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    #---------------------------------------------------------------------#
    "confidence"        : 0.5,
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    #---------------------------------------------------------------------#
    #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
    #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
    #---------------------------------------------------------------------#
    "letterbox_image"   : False,
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"              : True,
}
```
3. 运行predict.py，输入  
```python
img/street.jpg
```
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。  

## 评估步骤 
### a、评估VOC07+12的测试集
1. 本文使用VOC格式进行评估。VOC07+12已经划分好了测试集，无需利用voc_annotation.py生成ImageSets文件夹下的txt。
2. 在yolo.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**  
3. 运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

### b、评估自己的数据集
1. 本文使用VOC格式进行评估。  
2. 如果在训练前已经运行过voc_annotation.py文件，代码会自动将数据集划分成训练集、验证集和测试集。如果想要修改测试集的比例，可以修改voc_annotation.py文件下的trainval_percent。trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1。train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1。
3. 利用voc_annotation.py划分测试集后，前往get_map.py文件修改classes_path，classes_path用于指向检测类别所对应的txt，这个txt和训练时的txt一样。评估自己的数据集必须要修改。
4. 在yolo.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**  
5. 运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

## Reference
https://github.com/qqwweee/keras-yolo3/  
https://github.com/Cartucho/mAP  
https://github.com/Ma-Dan/keras-yolo4  
