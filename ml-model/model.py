# import pandas as pd
# from sklearn.naive_bayes import GaussianNB
# import pickle
# from sklearn.model_selection import train_test_split

# df = pd.read_csv('ml-model/data/Training.csv')

# X = df.iloc[:,:-1]
# y = df.iloc[:, -1]
# X_train, X_test, y_train, y_test =train_test_split(
# X, y, test_size = 0.2, random_state = 24)


# # y = df['prognosis']
# # X = df['symptoms']

# nb_model = GaussianNB()
# nb_model.fit(X_train, y_train)



# pickle.dump(nb_model,open('model.pkl','wb'))
