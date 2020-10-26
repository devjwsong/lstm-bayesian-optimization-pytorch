# lstm-bayesian-optimization-pytorch
This is a simple application of LSTM to text classification task in Pytorch using **Bayesian Optimization** for hyperparameter tuning.

The dataset used is *Yelp 2014* review data[[1]](#1) which can be downloaded from [here](http://www.thunlp.org/~chm/data/data.zip).

Detailed instructions are explained below.

<br/>

---

### Configurations

You can set various hyperparameters in `src/constants.py` file.

The description of each variable is as follows.

Note that for Bayesian Optmization, the hyperparameter to be tuned should be passed in a form of `tuple`.

So you can set an argument as a `tuple` or a certain value.

The former means that the argument will be included as the subject of Bayesian Optimization and the latter means that it should not be included.

<br/>

Argument | Type | Description | Default
---------|------|---------------|------------
 `device`         | `torch.device`                    | The device type. (CUDA or CPU)                               | `torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')` 
 `learning_rates` | `tuple (float, float)` or `float` | The range of learning rates. (or a value)                    | `(0.0001, 0.001)`                                            
 `batch_sizes`    | `tuple (int, int)` or `int`       | The range of batch sizes. (or a value)                       | `(16, 128)`                                                  
 `seq_len`        | `tuple (int, int)` or `int`       | The range of maximum sequence lengths. (or a value)          | `512`                                                        
 `d_w`            | `tuple (int, int)` or `int`       | The range of word embedding dimensions. (or a value)         | `256`                                                        
 `d_h`            | `tuple (int, int)` or `int`       | The range of hidden state dimensions in the LSTM. (or a value) | `256`                                                        
 `drop_out_rate`  | `tuple (float, float)` or `float` | The range of drop out rates. (or a value)                    | `0.5`                                                        
 `layer_num`      | `tuple (int, int)` or `int`       | The range of LSTM layer numbers. (or a value)                | `3`                                                          
 `bidirectional`  | `bool`                            | The flag which determines whether the LSTM is bidirectional or not. | `True`                                                       
 `class_num`      | `int`                             | The number of classes.                                       | `5`                                                          
 `epoch_num`      | `tuple (int, int)` or `int`       | The range of total iteration numbers. (or a value)           | `10`                                                         
 `ckpt_dir`       | `str`                             | The path for saved checkpoints.                              | `../saved_model`                                             
 `init_points`    | `int`                             | The number of initial points to start Bayesian Optimization. | `2`                                                          
 `n_iter`         | `int`                             | The number of iterations for Bayesian Optimization.          | `8`                                                          

<br/>

<hr style="background: transparent; border: 0.5px dashed;"/>

### How to run

1. Install all required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

2. Download the dataset and extract it.

   Of course, you can use another text classification dataset but make sure that the formats/names of files are same as those of *Yelp 2014* review dataset. (See the next step.)

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

<a id="1">[1]</a>  *Yelp Open Dataset*. ([https://www.yelp.com/dataset](https://www.yelp.com/dataset))

---



