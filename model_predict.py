from sklearn.externals import joblib
import preprocessing
import numpy as np

def model(q,ra,sa,mtype):
    # # add preprocessing calls here
    #
    # # load the saved model and return prediction
    # pred = ""
    q_basic = ['q_word_count','q_char_count','q_avg_word']
    a_basic = ['r_word_count','r_char_count','r_avg_word','s_word_count','s_char_count','s_avg_word']
    q_pos_basic = ['q_nouns','q_adjectives','q_verbs']
    q_pos_adv = ['q_nouns_vs_length','q_adjectives_vs_length','q_verbs_vs_length',
                 'q_nouns_vs_words','q_adjectives_vs_words','q_verbs_vs_words']
    a_pos_basic = ['r_nouns','r_adjectives','r_verbs','s_nouns','s_adjectives','s_verbs',]
    a_pos_adv = ['r_nouns_vs_length','r_adjectives_vs_length','r_verbs_vs_length',
                 'r_nouns_vs_words','r_adjectives_vs_words','r_verbs_vs_words',
                 's_nouns_vs_length','s_adjectives_vs_length','s_verbs_vs_length',
                 's_nouns_vs_words','s_adjectives_vs_words','s_verbs_vs_words']
    similarity = ['Jaccard','bm25']
    rouge1 = ['r1_f','r1_p','r1_r']
    rouge2 = ['r2_f','r2_p','r2_r']
    rougel = ['rlcs_f','rlcs_p','rlcs_r']
    new_pos1 = ['s_verbs_vs_r_verbs','s_nouns_vs_r_nouns','s_adjectives_vs_r_adjectives',
                's_word_count_vs_r_word_count','s_nouns_vs_words_vs_r_nouns_vs_words',
                's_verbs_vs_words_vs_r_verbs_vs_words','s_adjectives_vs_words_vs_r_adjectives_vs_words']
    new_pos2 = ['rs_word_diff','rs_noun_vs_words_diff','rs_verb_vs_words_diff','rs_adjectives_vs_words_diff']
    ibm_feat = ['precision','recall','F1_score']
    q_tags = ['how_flag','what_flag','why_flag','who_flag','which_flag','when_flag','where_flag','whom_flag']

    features  = q_basic + a_basic + q_pos_basic + q_pos_adv + a_pos_adv + similarity + rouge1 + rouge2 + rougel + new_pos1  + ibm_feat + q_tags

    columns = ['question','ref_answer','stu_answer']

    temp = get_basic_features(data,columns)

    cleaning_tasks = ['lemma','num']
    temp = clean(temp,cleaning_tasks,columns)

    temp = get_basic_POS(temp,columns)
    temp = get_advanced_POS(temp,columns)

    sim_columns = ['ref_answer','stu_answer']
    temp = get_Jaccard(temp,sim_columns)

    temp['bm25'] = 0

    scores = get_Rogue(temp,sim_columns)
    r1 = pd.DataFrame(scores)['rouge-1'].apply(pd.Series)
    r2 = pd.DataFrame(scores)['rouge-2'].apply(pd.Series)
    r3 = pd.DataFrame(scores)['rouge-l'].apply(pd.Series)
    r = pd.concat([r1,r2,r3] , axis = 1, )
    r.columns = ['r1_f','r1_p','r1_r','r2_f','r2_p','r2_r','rlcs_f','rlcs_p','rlcs_r']
    temp = pd.concat([temp,r],axis = 1)

    temp = get_new_POS1(temp)
    temp = get_new_POS2(temp)

    temp['precision'] = 0
    temp['recall'] = 0
    temp['F1_score'] = 0

    temp = get_question_tags(temp)

    temp.drop(['question','ref_answer','stu_answer'], axis=1, inplace=True)
    temp = temp[features]

    inp_feat = np.array(temp)
    if mtype == "classifier":
        loaded_model = joblib.load("classifier.sav")
        pred = loaded_model.predict(inp_feat).ravel()[0]
        if pred >= 0.5:
            return "Correct"
        else:
            return "Incorrect"
    else:
        loaded_model = joblib.load("regressor.sav")
        pred = loaded_model.predict(inp_feat).ravel()
        return str(pred)
    return "Correct"
