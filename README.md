# SentenceClassifier_ipavlov
iPavlov library based cnn sentence classifier

Use 'run' method to perform classification:

        ic = IntentsClassifier(root_config_path='root/cf_config.json')
        mes = input()
        print(ic.run(mes))
        
The result is a nested dictionary containing the decisions and confidence levels for both root and sub-categories.


Use 'train' method to train new model. Parameters for 'train':

- model_level - model level, 'root' or 'subs'.

- model_name - subcategory name. Set to '' for root model.

- path_to_data - path to training data. It should be stored in csv format with 'text' and 'labels' columns.

- path_to_config - path to config json file.

- test_size - fraction of data to use in hold-out dataset (default value: 0.15).

- aug_method - the way of augmenting training data (not applied for test, default value: 'word_dropout'). Set samples_per_class to None to disable data augmentation.

- samples_per_class number of samples per class in equalized dataset, None for leaving the classes distribution intact (default value: None).

- path_to_save_file - path to file to store the obtained weights.

- path_to_resulting_file - path to copy the best weights obtained during training.


Use 'get_performance' method to evaluate model on test set with f1 metric (macro averaging). Called automatically at the end of 'train'.
