## vectorprob: Estimating conditional word probabilities from pretrained static word vectors

Suppose we want to know the distribution p(w | c) where w and c are both single words, in some certain relationship. This software uses pretrained static word embeddings to estimate such distributions.

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





