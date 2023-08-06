# embed2word
Turning natural language text into numerical representations has been a focus of NLP for quite some time. The idea is to represent each word with a vector that encapsulates the word's meaning. In 
the past, word vectors like GLOVE proved useful, although they did not capture "context". With the advent of large language models like GPT, contextual word representations became more prevalent. 
Here we intend to use the GPT2 model to represent word vectors. This work was done previously, based on a conversation in 2019: 
https://github.com/huggingface/transformers/issues/1458#issuecomment-1564253345.

Manipulating word embedding vectors and then converting those vectors back to words is known as semantic arithmetic. Typically, word vectors are low dimensional representations of tokens, which 
are not necessarily invertible. Here we are using GPT2 as the base model, so having the GPT2LMHeadModel and GPT2Tokenizer is necessary. The PyTorch library is also used.

## process
The idea is to start from text where the goal is to replace one sentiment with another. In order to do that, we first turn all text and sentiment words into vectors. We perform the arithmetic 
operations, and then project back to the word vocabulary. GPT2 has a limited vocabulary of around 50K, which is significantly less than GPT3's 14,735M vocabulary. Therefore, we do not 
expect it to perform well, and it does not.


## example 
$ python test_embed2vec.py "This was a good resturant. Their ramen is great." "great" "horrible"
This was horrible horrible resturant. Their ramen is horrible.

 
