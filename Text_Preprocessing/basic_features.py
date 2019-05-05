def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))


def get_basic_features(df,columns):
  
  for column in columns:
    df[column[0]+'_word_count'] = df[column].apply(lambda x: len(str(x).split(" ")))
    print(column+"Word Count Done")
    df[column[0]+'_char_count'] = df[column].str.len()
    print(column+"Char Count Done")
    df[column[0]+'_avg_word'] = df[column].apply(lambda x: avg_word(x))
    print(column+"Avg Word Length Done")

  return df
