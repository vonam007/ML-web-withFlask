 ----------------------------
 run this in Terminal to activate the virtual environment:

 .venv\Scripts\activate

 ----------------------------
if  don't have flask, run this in terminal:

pip install flask

-----------------------------
cd to FLASK_APP folder, run this to run app:

flask --app main run

-----------------------------
press ctrl+C to stop. 
If there are any change, re-run the app.

-----------------------------   
try demo our app: https://vonam007.pythonanywhere.com/   

-----------------------------
###How to use model:###   
Step 1: check GPU by run check_gpu.py. You need to have GPU for training model   
(Or use my pretrained with 67 epochs and reach loss: 0.08 - https://1drv.ms/u/s!Atb46kl_Ra3rjipFvchBwKyOUSGu?e=GLpuGi)   
Step 2: run train_model.py   
Step 3: run predict.py for prediction or run evaluation.py to evaluate model   
-----------------------------
Division of work:   
Trần Quang Nhật - 20520675: Trained model, Evaluated model, Searched related documents   
Võ Thành Thái - 20520305: Searched dataset, built website interface and functions   
Võ Nguyễn Hoài Nam - 20520645: Deploy the website, converted h5 model to json, loaded model to website
