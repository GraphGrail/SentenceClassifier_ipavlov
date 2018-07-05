# SentenceClassifier_ipavlov
iPavlov library based cnn sentence classifier

Use 'run' method to perform classification:

        ic = IntentsClassifier(root_config_path='root/cf_config.json')
        mes = input()
        print(ic.run(mes))
        
The result is a nested dictionary containing the decisions and confidence levels for both root and sub-categories.
