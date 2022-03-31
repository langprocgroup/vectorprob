# vectorprob: Estimating conditional word probabilities from pretrained static word vectors

Suppose we want to know the distribution p(w | c) where w and c are both single words, in some certain relationship. This software uses pretrained static word embeddings to estimate such distributions.

The Python script `bilinear.py` will train the probability model based on example counts of (w,c) pairs, saving a model in `.pt` format. Then the script `run_model.py` will apply the model to get conditional probabilities for a new set of (w,c) pairs.

```
usage: bilinear.py [-h] [--dev DEV] [--test TEST] [--vocab VOCAB] [--tie_params] [--softmax] [--num_iter NUM_ITER] [--batch_size BATCH_SIZE] [--lr LR] [--activation ACTIVATION] [--structure STRUCTURE] [--no_encoders] [--dropout DROPOUT] [--check_every CHECK_EVERY] [--patience PATIENCE] [--output_filename OUTPUT_FILENAME] [--include_unk] [--data_on_device]
                   [--seed SEED] [--finetune] [--weight_decay WEIGHT_DECAY] [--batch_norm] [--layer_norm]
                   vectors train

Estimate conditional word probabilities using a log-bilinear model

positional arguments:
  vectors               Path to word vectors in word2vec format
  train                 Path to file containing training counts of word pairs

optional arguments:
  -h, --help            show this help message and exit
  --dev DEV             Path to file containing dev counts of word pairs
  --test TEST           Path to file containing test log probabilities
  --vocab VOCAB         Limit output vocabulary to words in the given file if provided
  --tie_params          Set phi = psi
  --softmax             Only use vectors for the context word, not the target word
  --num_iter NUM_ITER
  --batch_size BATCH_SIZE
                        batch size; 0 means full gradient descent with no batches
  --lr LR               starting learning rate for Adam
  --activation ACTIVATION
                        activation function for networks
  --structure STRUCTURE
                        network structure, same for phi and psi
  --no_encoders         Do not train word encoders. Overrides structure arguments.
  --dropout DROPOUT     dropout
  --check_every CHECK_EVERY
                        Record progress and check for early stopping every x iterations
  --patience PATIENCE   Allow n increases in dev loss for early stopping. None means infinite patience. Default None.
  --output_filename OUTPUT_FILENAME
                        Output filename. If not specified, a default is used which indicates the time the training script was run..
  --include_unk         Include UNK target words in dev and test sets.
  --data_on_device      Store training data on GPU (faster for big datasets but uses a lot of GPU memory).
  --seed SEED           Random seed for minibatches.
  --finetune            Finetune word vectors.
  --weight_decay WEIGHT_DECAY
                        Weight decay.
  --batch_norm          Apply batch normalization.
  --layer_norm          Apply layer normalization.
  ```

## Example

Suppose you have word vectors in word2vec format in a file `vectors.vec`, and a file of counts of word pairs in csv format at `train_counts.csv`. This file should be formatted such that the first column is the target word, the second column is the context word, and the third column is the count. You might also want to use a second file of counts, called `dev_counts.csv`, which will be used to evaluate training progress; and a file `vocab.txt` which contains a list of words (one per line) serving as the target word vocabulary (the support of the distribution p(w|c)). 

Then you can train the model with a command such as:
```
python bilinear.py vectors.vec train_counts.csv --output_filename model.pt --dev dev_counts.csv --vocab vocab.txt --num_iter 5000 --batch_size 512 --structure "[300, 400, 400]" --data_on_device --layer_norm 
```
This will train the model and save it as `model.pt`. The model is trained for 5000 steps of gradient descent with batch size 512; the word and context encoders have structure `"[300, 400, 400]"` meaning the input has 300 dimensions, there is a hidden layer of 400 dimensions, and the final output has 400 dimensions; layer normalization is applied through the option `--layer_norm`. The script will output statistics including train set and dev set loss. The most important number to watch is the dev set loss; the model is most accurate when this is lowest.

Once the model is trained, if you have a set of pairs you want to get probabilities for in csv format (say `test.csv`), you can run the model as so:
```
python run_model.py model.pt test.csv
```
This will output the log probabilities for the target words to `stdout` in csv format.