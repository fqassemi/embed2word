from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F


gpt_tokenizer = GPT2Tokenizer.from_pretrained('/HuggingFace/models/gpt2/')
gpt_model = GPT2LMHeadModel.from_pretrained('/HuggingFace/models/gpt2/')  # or any other checkpoint
word_embeddings = gpt_model.transformer.wte.weight  # Word Token Embeddings 

vocab_list = []
for i in range(gpt_tokenizer.vocab_size):
    vocab_list.append(gpt_model.transformer.wte.weight[[i],:][0] )

vocab_tensor = torch.stack(vocab_list, dim=0).squeeze(1)

def sentiment_algebra(inp_str, sent1, sent2):
#    import spacy
#    nlp = spacy.load('en_core_web_sm')
#    sents = nlp(inp)
    sents = [inp_str]
    sent = ""
    for inp in sents:
        text_index_inp = gpt_tokenizer.encode(inp, add_prefix_space=True)
        vector_inp = word_embeddings[text_index_inp,:]
        #print(vector_inp.shape)

        text_index_sent1 = gpt_tokenizer.encode(sent1, add_prefix_space=True)
        vector_sent1 = word_embeddings[text_index_sent1,:]
        #print(vector_sent1.shape) 

        text_index_sent2 = gpt_tokenizer.encode(sent2, add_prefix_space=True)
        vector_sent2 = word_embeddings[text_index_sent2,:]
        #print(vector_sent2.shape) 

        new_inp_vec = vector_inp - torch.mean(vector_sent1, dim=0) + torch.mean(vector_sent2, dim=0)
        #print(new_inp_vec.shape)

        cos_sim = []
        for i in range(new_inp_vec.shape[0]):
            cos_sim.append(F.cosine_similarity(new_inp_vec[i], vocab_tensor, dim=1))

        for i in range(len(cos_sim)):
            ind = torch.argmax(cos_sim[i])
            sent += gpt_tokenizer.decode([ind], add_prefix_space=True)
    
    return sent    
