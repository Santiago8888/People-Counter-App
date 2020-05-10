# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

**The process behind converting custom layers involves...**

**Disclaimer:** It's important to remark that converting custom layers depends on the ML framework used, I will proceed with the explanation for **TensorFlow.**

The first step is to buld the model, during this process the inputs and targets are defined for the TF model and the layers, incluiding the custom ones are attached to the Network Builder,
during this steps desired optimization and accuracy are set. The returned value is a checkpoint file that we can optimize with OpenVino toolkit.

After that, we will need a different handlers for each extension (CPU, GPU, etc.) and for the model optimizer, we can generate templates using the extension generator.

The first template, is the **extractor** this file specifies a diffent treatment for each custom layer.
This file may not require modification and rather depends on the names given to the custom layers.

Additionally, for each layer we need to set an Operation Extension File, thorugh this file we could skip the custom layer (if we decided to) 
given that the output shape remains the same we set the class property enabled to false. 
In the case that the layer is enabled, we define the transformation of the shape (if required) during the computation of the layer.
To create the IR files for the Inference Engine, besides the usual command we need to specify the custom layers to be included this will generate the regular .xml and -bin files.

The final step, is to encode the execution rules for the layer in C++ to resemble the steps taken by Tensofr Flow on the graph. 
The template for these handlers were also generated during the extension generator step. 

Given the computation is executed without error we are now be able to request predictions from our new optimized model.


**Some of the potential reasons for handling custom layers are...**

1. Creating a community, similar to how the Open Source thrives.
2. Keep up pace with recent development, research and innovation in Deep Learning. 
3. Set the foundations to support other frameworks.
4. 80-20 rule, provide support to the 20% most common layers, will support 80% of the models.
5. Give developers an opportuity to write their first lines of C (personal experience). 


## Comparing Model Performance

**My method(s) to compare models before and after conversion to Intermediate Representations were...**
I ran the frozen .pb file using TensorFlow on the same computer that the Intel model ran.
The script is located at scripts/benchamark.py 
As isntructwed the original model was not included.
The script was run locally. 

**The difference between model accuracy pre- and post-conversion was...**
For the object classifier the difference was **less than 2% ** while for the detection box coordinates it was slightly above 2%.
Raw Outputs are stored on the benchmarks.txt


**The size of the model pre- and post-conversion was...**
The optimized model was 2MB smaller or around 3% than the original frozen model.
This calculation does not include a comparisson between Tensorflow and its dependencies or OpenVino's.
Although, the difference in size is not big this is unimportant as storage size is cheap including for IoT device.

**The inference time of the model pre- and post-conversion was...**
This is the nice part: The difference was **100 times faster.**  While on average the optimized model infers every 30ms. The original tensorflow took more than 3 seconds for each request. 
This is difference between the ability to proccess video in real-time and not.

I want to include an additional metric on the benchamrk, and is the develper time. 
It took me roughly half the time to run the optimized model than the TensorFlow.
OpenVino comes with preset abstractions that allow developers to get results without going in depth to the model's documentation.

**OpenVino is quickly becoming my prefered framework to rapidley test models and deploy solutions.**


## Assess Model Use Cases

Some of the potential use cases of the people counter app are listed in:

[4 Business Models for a People Counter App at the Edge.](https://medium.com/@santiagomartnez_69416/4-business-models-for-a-people-counter-app-at-the-edge-4c48b9f8e0c0)



## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

- **Lighting** depends on the application and its environment. 
As its not the same to deploy an app outdoors as inddors, even outdoors there might not be a need to run the inference during a good portion of the night.
A good solution for this, is that after testing a prototype and identifying the most error prone illumniation settings augment (artificially) the dataset and retrain the model.
Other solutions for this is preprocessing the image, this is relatively easy on IoT applications as security cameras since they are expected to be static.
This opens the door for using ensamble learning for example on models trained on black & white images.

- **Model Accuracy** is also dependant on the application. 
In particular, is important to optimize based on the Confussion Matrix for the application and what errors are most costly (False Postive or False Negative).
Applications as security, medical and financial may have less tolerance for False Negatives as opposed to Telematics, Weather monitoring or even business aplications.
Is a good practice that once deployed the model to send randmo batches of images (or data) close to the threshold to guarantee they are accurate and properly set.
Dashboards and domain experts are good herlpers for this Quality Assurance process.

- **Image Size**, afects the performance and accuracy of the model. 
This is one of the factors that is better to account from the beggining, as the model training and dataset depends on it.
Big images can be give a precise accuracy but have the withdraw that may take long to execute. In particular small images are better for video dependant applications.
Transparent communication with stackholders and explorations of the available literature may improve the chances of getting the image size right from the beggining.

- **Focal Length**, is better taken into account during the production and deployment of the solution and is better handled by the operations team rather than development.
In particular accuracy is a wild card to adjust the images to fit the input to the model before the image presproccesing steps.
As a developer, unless there is a clear use case where a non conventional focal length is to be employed I would stick to most conventionals.

