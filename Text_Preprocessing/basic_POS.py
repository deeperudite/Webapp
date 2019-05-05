def tag_part_of_speech(text):
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    pos_list = pos_tag(text_splited)
    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])
    adjective_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])
    verb_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])
    return[noun_count, adjective_count, verb_count]
  
def get_basic_POS(df,columns):
    
    for column in columns: 
      print("Generating Basic POS Features for "+ column)
      df[column[0]+'_nouns'], df[column[0]+'_adjectives'], df[column[0]+'_verbs'] = zip(*df[column].apply(lambda comment: tag_part_of_speech(comment)))
      
    return df  
