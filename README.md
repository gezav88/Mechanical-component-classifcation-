# component_classification

repository for component classification from 3D objects
NCSU Course: Smart Manufacturing Course:
+ Author: Gerardo Zavala.
+ Professor: Binil Starly

Python 3.8.5

Repo Strucutre: 

+ Folders: 

* data: contains the info of the dictionaries to map from class numeric to name of class.

* models: saved models from training. 
    * model_v1
    * model_v2

* notebooks: contains the notebooks that were used to explore the data and train the models.
  * Mechanical_Components_project (notebook to train and evaluate)

* static: contains the style for the app. 
  *  style.css

* templates: template for the html web app. 
  * index.html

* utils: contains all the functions needed to run the app and deploy. 
  * util_funcs.py

+ Files:
* requirements.txt (information of libraries needed to run the code). 
* app.py

 
*****************************************************************************************************************************************************************

## Steps to run: (local)
after training the model in google colab, save the models in models folder,  models will be loaded from there. 

$ python app.py 
a local server is open and it can be used to predict based on the selected file to upload to uploads and then selecting the file from that folder. 


Once the app is deployed local:

1.-Click the upload button.

2.- Select a file to clasify from the dataset/test/   and hit submit    (Remember files will be uploaded into uploads folder in the repo folder)

3.- Now selecct file button below submit and select one of the files you upload in the uploads folder

4.- click classify object from file (blue button) 

