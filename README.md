# lstm-bayesian-optimization-pytorch
This is a simple application of LSTM to sentiment classification task in Pytorch using **Bayesian Optimization** for hyperparameter tuning.

The dataset used is *Yelp 2014* review data[[1]](#1) which can be downloaded from [here](http://www.thunlp.org/~chm/data/data.zip).

Detailed instructions are explained below.

<br/>

---

### How to run

1. Install all required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

2. Download the dataset and extract it.

   <br/>

3. Make a directory named `data`.

   Get files named `train.txt`, `text.txt`, `dev.txt` and `wordlist.txt` from `yelp14`  and put them into `data`.

   The directory structure should be as follows.

   - data
     - train.txt
     - test.txt
     - dev.txt
     - wordlist.txt

   <br/>

4. Execute below command to train the model.

   ```shell
   python src/main.py --mode='train'
   ```

   - `--mode`: This specify the running mode. The mode can be either `train` or `test`.

   <br/>

   The Bayesian Optimization is used for hyper-parameter tuning in this task.

   You can add/modify the hyperparameter list to tune in `main.py`.

   ```python
   self.pbounds = {
       'learning_rate': learning_rates,
       'batch_size': batch_sizes
   }
   
   self.bayes_optimizer = BayesianOptimization(
       f=self.train,
       pbounds=self.pbounds,
   	random_state=777
   )
   ```

   Currently, the batch size and the learning rate are only subjects to be adjusted.

   If you want to modify `self.pbounds`, add the desired hyperparameter and change its value in `constant.py` into a tuple consisting of two values, minimum and maximum, sequentially.

   Then you should add that hyperparameter as an additional parameter for the function `train` like `batch_size` and `learning_rate`.

   <br/>

5. After training, you can test the model with test data by following command.

   ```shell
   python src/main.py --mode='test' --model_name=MODEL_NAME
   ```

   - `model_name`: This is the file name of trained model you want to test. The model is located in `saved_models` directory if you didn't change the checkpoint directory setting.

<br/>

---

### References

<a id="1">[1]</a> 
*Yelp Open Dataset*. ([https://www.yelp.com/dataset](https://www.yelp.com/dataset))

---



