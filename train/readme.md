# IDIL: Imitation Learning of Intent-Driven Expert Behavior

## Dataset
* Please unzip `train/data.zip` to use *MultiGoals*, *OneMover*, and *Movers* datasets.

* The dataset for *AntPush-v0* can be found at this link: [link](https://github.com/id9502/Option-GAIL/tree/main/data/mujoco).
Please convert the format of this dataset using the following command::
  ```
  python3 train/dataconv.py --data-path=<path-to-torch-data> --save-path=<path-to-save> --clip-action=True
  ``` 

* The datasets for other *Mojoco* domains can be found at this link: [link](https://github.com/Div99/IQ-Learn/tree/main/iq_learn/experts).


## Examples
* Please check the `train/script_examples/` folder to find example scripts. 
* To reproduce the paper results, please check hyperparameters specified in Appendix C.5.
* Please check if the expert data path is correctly configurated in the `train/conf/` folder.


