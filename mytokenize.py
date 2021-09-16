
# define a my own Tokenizer class

class Tokenizer : 
    def __init__(self, spm_model) : 
        self.spm_model = spm_model
        self.bos = "<s>"
        self.eos = "</s>"
        self.unk = "<unk>"
        self.eos_id = self.spm_model.piece_to_id(self.bos)
        self.bos_id = self.spm_model.piece_to_id(self.eos)
        self.unk_id = self.spm_model.piece_to_id(self.unk)

    def encode(self, sentence, max_len = None) : 
        input_ids = self.spm_model.encode_as_ids(sentence)
        attention_mask = [1] * len(input_ids)

        if max_len : 
            pad = (max_len - len(input_ids))
            input_ids = input_ids + [0] * pad
            attention_mask + [0] * pad

        encoding = {"input_ids" : input_ids, "attention_mask" : attention_mask}

        return encoding
        
        return 

    def encode_plus(self, sentence, max_len = None) :
        if not isinstance(sentence, str) : 
            raise TypeError

        encoding = self.encode(sentence)
        input_ids = [self.bos_id] + encoding["input_ids"] +  [self.eos_id]
        attention_mask = [1] * len(input_ids)

        if max_len : 
            if max_len > len(input_ids) : 
                pad = (max_len - len(input_ids))
                input_ids = input_ids + [0] * pad
                attention_mask + [0] * pad
            else : 
                input_ids = input_ids[:max_len]
                attention_mask = attention_mask[:max_len]

        encoding = {"input_ids" : input_ids, "attention_mask" : attention_mask}

        return encoding

    def decode(self, input_ids) : 
        if not isinstance(input_ids, list) : 
            raise TypeError
        
        if input_ids[0] == self.bos_id  :  
            input_ids = input_ids[1:]
        
        if input_ids[-1] == self.eos_id: 
            input_ids[:-1]

        if input_ids[-1] == 0 : 
            zero_indx = input_ids.index(0)
            input_ids = input_ids[:zero_indx]

        decode = self.spm_model.decode_ids(input_ids)
        return  decode

    def decode_plus(self, input_ids) : 
        return ""