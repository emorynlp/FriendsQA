# Hierarchical Transformer for Span-based QA

## Citation

* [Transformers to Learn Hierarchical Contexts in Multiparty Dialogue for Span-based Question Answering](). Changmao Li and Jinho D. Choi. In Proceedings of the Conference of the Association for Computational Linguistics, ACL'20, 2020.
* Note that some of the source codes are based on [`huggingface/transformers`](https://github.com/huggingface/transformers).


## Source files include:

### Data process files:

```
/src/transformers/data/processors/umlm.py 
```

* Read generated utterance-level masked language modeling data into examples and create features from the examples

```
/src/transformers/data/processors/uop.py 
```
* Read generated utterance order prediction (UOP) data into examples and create features from the examples

```
/src/transformers/data/processors/friendsqa.py 
```

* Read friendsqa data into examples and create features from the examples


### Model files:

```
/src/transformers/modeling_bert.py:
```

* Token-level masked language modeling BERT model 
* Utterance-level masked language modeling BERT model 
* Utterance order prediction BERT model 
* QA whole context fine-tuning BERT model  
* QA split context fine-tuning BERT model

```
/src/transformers/modeling_roberta.py:
```

* Token-level masked language modeling RoBERTa model 
* Utterance-level masked language modeling RoBERTa model 
* Utterance order prediction RoBERTa model 
* QA whole context fine-tuning RoBERTa model  
* QA split context fine-tuning RoBERTa model 


### Executive files:

```
/src/examples/run_language_modeling.py
```

* Run BERT or RoBERTa token-level masked language modeling(TMLM)

```
/src/examples/run_umlm.py
```

* Run BERT or RoBERTa utterance-level masked language modeling(UMLM)

```
/src/examples/run_uop.py 
```

* Run BERT or RoBERTa utterance order prediction(UOP)

```
/src/examples/run_friends_whole_context.py 
```

* Run BERT or RoBERTa fine-tuning on FriendsQA in whole context format

```
/src/examples/run_friends_split_context.py 
```

* Run BERT or RoBERTa fine-tuning on FriendsQA in split context into utterances format


### Other utility files:

```
/src/utils/test_models.py
```

* Test if all neural models correctly work

```
/src/utils/categorizing.py
```

* Categorize results by question types

```
/src/utils/analysis.py
```

* Analyze categorized results

```
/src/utils/evaluate_whole_context.py
```

* Evaluate the whole context QA fine-tuning results 

```
/src/utils/evaluate_split_context.py
```

* Evaluate the split context QA fine-tuning results


## Other Notices

You need to generate your own language model pre-training data fit for your own corpus.
Here is the data format for language model pre-training data.

### Pre-training data format

#### TMLM 

* The TMLM data format is a text file.
* Each line is an utterance and there is an empty line between dialogues.

#### UMLM

* The UMLM data format is a csv file.
* Each line includes a tokenized one token masked utterance and the masked token. The separator is \t.
* You can modify it to your own format in /src/transformers/data/processors/umlm.py if you want.

#### UOP

* The UOP data format is a json file.
* The json format is 

   ```python
   [
       {
         "utterances": ["u1", "u2", .....] # The utterances list, 		
         "is_correct_order": "Yes" or "No" # If the utterance is in correct order or not
       },
       ....
   ]
   ```

* You can modify it to your own format in /src/transformers/data/processors/uop.py if you want.

