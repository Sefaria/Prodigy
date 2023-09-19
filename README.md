# Prodigy
Repo for building Prodigy (www.prodi.gy) server.

## Install DBManager locally

To install the DBManager plugin for Prodigy which allows Prodigy to interface with MongoDB, follow the instructions [here](https://github.com/Sefaria/Prodigy/blob/main/prodigy_utils/README.txt)

## Build Prodigy Docker image

All resources for building a prodigy Docker image are in `build/prodigy`. 

## Build Prodigy StatefulSet

Resources for building a kubernetes StatefulSet for hosting Prodigy are in `build/prodigy/annotator`. To change the settings for the Prodigy server when it starts up, change the `args` passed to the `prodigy` container in `build/prodigy/annotator/templates/prodigy.yaml`. Below is documentation on the args available:

### ner-recipe Documentation

The `ner-recipe` function is a Prodigy natural language processing recipe to train Spacy models based on Named Entity Recognition tasks from a MongoDB database.

#### Parameters

Here are the parameters for the `ner-recipe` function:

- **dataset** (str, positional): Name of the dataset where answers will be saved.

- **input-collection** (str, positional): Name of the MongoDB collection from where data will be inputted.

- **output-collection** (str, positional): Name of the MongoDB collection where output will be stored.

- **labels** (str, positional): Labels used for the classification task. Comma separated with no spaces.

- **model-dir** (str, optional): The location of the Spacy model in your directory. Use if you want to have Prodigy pre-tag input to fine-tune an existing model.

- **view-id** (str, optional): Specifies the annotation interface to use in this recipe. See [here](https://prodi.gy/docs/recipes/#ner) for a full list of options. For NER, best is to use `ner_manual`.

- **db-host** (str, optional): Specifies the MongoDB host.

- **db-port** (int, optional): Specifies the MongoDB port.

- **user** (str, optional): Specifies the MongoDB username.

- **replicaset-name** (str, optional): Specifies the MongoDB Replicaset Name.

- **dir** (str, optional): Direction of text to display. Either 'ltr' or 'rtl'.

- **lang** (str, optional): Lang of training data. Either 'en' or 'he'.

- **should-add-predictions** (int, optional): When there is an existing model, should you use it to add predictions to input. Either `1` or `0`.

####

Below is an example command to run prodigy

```
prodigy ner-recipe ref_tagging webpages_en_input webpages_en_output Citation,Person -should-add-predictions 1 -model-dir /prodigy-disk/webpages_en -lang en -dir ltr --view-id ner_manual -db-host localhost -db-port 27017
```
