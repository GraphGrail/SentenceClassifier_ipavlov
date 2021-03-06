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

- test_size - fraction of data to use in hold-out dataset (default value: 0.15)

- aug_method - the way of augmenting training data (not applied for test, default value: 'word_dropout'). Set samples_per_class to None to disable data augmentation.

- samples_per_class number of samples per class in equalized dataset, None for leaving the classes distribution intact (default value: None).

- path_to_global_embeddings - path to embeddings file in fasttext '.bin' format.

- path_to_save_file - path to folder to store the weights obtained during training.

- path_to_resulting_file - path to folder to store the best weights after training (the last saved weights file will be copied to this folder).

For example:

	ic.train(model_level='root',
         model_name= '',
         path_to_data='../ai_models_train/42/df_raw.csv',
         path_to_config='../ai_models_train/42/cf_config_dual_bilstm_cnn_model.json',
         path_to_global_embeddings='../ai_models/shared/ft_native_300_ru_wiki_lenta_lemmatize.bin',
         samples_per_class=1500,
         class_names=['доставка', 'оплата', 'другое', 'намерение сделать заказ'],
         path_to_save_file='../ai_models_train/42/',
         path_to_resulting_file='../ai_models_train/42/')

             
Use 'get_performance' method to evaluate model on test set with f1 metric (macro averaging). Called automatically at the end of 'train':

        perf = self.get_performance(path_to_config, model_path+'df_test.csv')

Use 'get_status' method to check if a particular model (specified via directory name) is currently training.

Use 'check_config' method to validate the config file for the model:



        from utils.check_config import check_config
        check_results = check_config(path_to_config)
        if len(check_results)>0:
            raise InvalidConfig(check_results,'Config file is invalid')


All model's files are stored in config['model_path'] folder. Other paths contain just filenames.

The model also logs its' performance with tensorboard. In order to retrieve the latest performace metric listed in the config['train']['metrics'] call get_latest_accuracy method with path to config file:

	IntentsClassifier.get_latest_accuracy('../ai_models_train/42/cf_config_dual_bilstm_cnn_model.json')
