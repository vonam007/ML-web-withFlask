 ----------------------------
 ### How to Use the Model ###

1. Activate the virtual environment by running the following command in the Terminal:

 `.venv\Scripts\activate`

*Note: This assumes you are using Windows. For Unix-based systems, use the command `source .venv/bin/activate`.*

2. If Flask is not installed, run the following command in the Terminal:

`pip install flask`

3. Change to the FLASK_APP folder in the Terminal:

`cd FLASK_APP`

4. To run the app, use the following command:

`flask --app main run`

Press `ctrl+C` to stop the application. If there are any changes, re-run the app.

### Demo Application ###

You can try out our demo application at [https://vonam007.pythonanywhere.com/](https://vonam007.pythonanywhere.com/). 

-----------------------------

### How to Use the Model ###

Follow these steps to use the model:

1. Check GPU availability by running `check_gpu.py`. Please note that a GPU is required for training the model.

2. Run `train_model.py` to train the model.

3. Use `predict.py` for making predictions with the trained model. Alternatively, you can run `evaluation.py` to evaluate the model.

-----------------------------
## Team Members and Tasks

| Team Member          | Tasks                                                    |
|----------------------|----------------------------------------------------------|
| Trần Quang Nhật      | Trained model, Evaluated model, Searched related documents |
| Võ Thành Thái        | Searched dataset, built website interface and functions   |
| Võ Nguyễn Hoài Nam   | Deployed the website, converted h5 model to json, loaded model to website |
