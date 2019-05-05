def clean(df,tasks,columns):
  
  for column in columns:
      #Lowercase conversion
      df[column] = df[column].apply(lambda x: x.lower())
      print(column+": Converted to lowercase")

      #Tokenization
      df[column] = df[column].apply(word_tokenize)
      df[column] = df[column].apply(lambda x: " ".join(x))
      print(column+": Tokenized")

      if('- ' in tasks):
        #Split a-b into a and b
        df[column] = df[column].str.replace('-',' ')
        print(column+": - Replaced")

      elif('-_' in tasks):
        #Split a-b into a and b
        df[column] = df[column].str.replace('-','_')
        print(column+": - Replaced")

      if('punct' in tasks):
        #Removing punctuations
        df[column] = df[column].str.replace('[^\w\s]',' ')
        print(column+": Removed punctions ")

      if('num' in tasks):
        #Replacing numbers
        df[column] = df[column].str.replace('[0-9]','#')
        print(column+": Replaced Numbers ")

      if('stop' in tasks):
        #Removing Stop Words
        df[column] = df[column].apply(lambda x: " ".join([w for w in x.split() if w not in stop_words]))
        print(column+": StopWords Removed")

      if('lemma' in tasks):
        #Lemmatization - root words
        df[column] = df[column].apply(lambda x: " ".join([lemmatizer.lemmatize(word,pos='v') for word in x.split()]))
        print(column+": Root words Lemmatized")

  print("Null values:",df.isnull().values.any())
  #print(df.head())
  return df
