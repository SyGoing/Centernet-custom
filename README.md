# Custom Centernet for face detection and other object detection tasks. 
   In this repo, you can training object detection tasks with many other backbones not only the backbones in this repo. 
   Since I am used to use the voc format dataset(the data is always marked by LabelImg), the voc format dataset has been provided. 
   Anyway, you could also use the centernet original coco format to train the model.
   For application, I have provided the python shell to convert the *.pth to onnx.

## Requirements for this repo
   python3.6 or higher
   Pytorch 1.1
   cocoapi 
   opencv-python
   Cython
   numba
   progress
   matplotlib
   easydict
   scipy
   
   Anaconda 3 is recommended
   
   
## Training 
   ### Wilderface Data Prepare
     Refered to the [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) , Generate the voc format wilderface dataset and copy it to the ./data 
	 And you can use it to train or generate the json format annotations for coco format 
	 
   ### Training wilderface with centerface's structure(relu) without keypoints
     in  the ./src/main.py : from opts_voc import opts  or from opts_coco import opts 
	 cd ./src/, python main.py 
	 
   ### Training wilderface with centerface's structure(relu) with keypoints
      * (1) heatmap way just like the original CenterNet's person keypoints
	  If you want to train the model with keypoints , you need to generate two json format annotations for coco format in the data/wilderface/ as follows:
	  wildertrain.json and wilderval.json by the VOC2COCOtrain.py and VOC2COCOval.py in the ./data dir .
      in  the ./src/main.py : from opts_hplm import opts
	  cd ./src/, python main.py 
	  
	  * (2) Just directly regress keypoints by coordinates
	   Actually,the Centerface's keypoints regression
	   in  the ./src/main.py : from opts_reglm import opts
	   cd ./src/, python main.py 
    
## Inference demo
	In the root_dir/src/demo.py , import relative opts(eg. from opts_voc import opts )
	python demo.py --demo [img|video|webcam] --load_model [model_path,../models/model_best.pth]  (just like the original centernet)
	
## Wilderface AP (easy ,medium and hard)
    root_dir/wilderface_eval_space 
	Python Evaluation Code for Wider Face Dataset refered to [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
	* run evaluation_on_widerface.py
	    import relative opts(eg. from opts_voc import opts ) in the evaluation_on_widerface.py
        python evaluation_on_widerface.py
		
    * before evaluating
        python setup.py build_ext --inplace
		
    * evaluating
      GroungTruth: wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat,wider_hard_val.mat
      python evaluation.py -p <your prediction dir> -g <groud truth dir>
	  
## Convert model to onnx
    import relative opts(eg. from opts_voc import opts ) in the evaluation_on_widerface.py 
	in the opts_voc.py ,--load_model (your trained model)
	root_dir/src/pytorch2onnx.py
	python pytorch2onnx.py , and the onnx model is in the onnxmodel dir
	
	For tensorRT's inference C++, you can refer to [TensorRT-CenterNet](https://github.com/CaoWGG/TensorRT-CenterNet)
	
	
## TODO
   - [] Some Efficient pretrained models and mAP report in the wilderface
   - [] Application demos by convert the onnx to other formats (mnn,ncnn,caffe for HUAWEI' nnie and npu) 
   
## Reference
   * Most refered to [CenterNet](https://github.com/xingyizhou/centernet) by[xingyizhou](https://github.com/xingyizhou)
   * [Centerface](https://github.com/Star-Clouds/CenterFace) by[Star-Clouds](https://github.com/Star-Clouds)  
   * [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) by[Linzaer](https://github.com/Linzaer)
   * [CenterFace.pytorch](https://github.com/chenjun2hao/CenterFace.pytorch) by[chenjun2hao](https://github.com/chenjun2hao)
   
   * [CenterMulti](https://github.com/bleakie/CenterMulti) by[bleakie](https://github.com/bleakie)
   
   * [TensorRT-CenterNet](https://github.com/CaoWGG/TensorRT-CenterNet) by[CaoWGG](https://github.com/CaoWGG)
   
   
   
   

	
   
