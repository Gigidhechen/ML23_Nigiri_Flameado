import numpy as np

class NaiveBayes():
    def __init__(self, alpha=1) -> None:
        self.alpha = 1e-10 if alpha < 1e-10 else alpha

    def fit(self, X, y):
        # TODO: Calcula la probabilidad de que una muestra sea positiva P(y=1)
        self.prior_positives = np.sum(y==1)/len(y)

        # TODO: Calcula la probabilidad de que una muestra sea negativa P(y=0)
        self.prior_negative = np.sum(y==0)/len(y)

        # TODO: Para cada palabra del vocabulario x_i
        # calcula la probabilidad de: P(x_i| y=1)
        # Guardalas en un arreglo de numpy:
        # self._likelihoods_positives = [P(x_1| y=1), P(x_2| y=1), ..., P(x_n| y=1)]
        self._likelihoods_positives = []
        for i in range(X.shape[1]):
            PosCount=np.sum((X[:,i]==1)&(y==1))
            PosLikelihood= (PosCount+self.alpha)/(np.sum(y==1)+2*self.alpha)
            self._likelihoods_positives.append(PosLikelihood)
        self._likelihoods_positives =  np.array(self._likelihoods_positives)
        
        # TODO:  Para cada palabra del vocabulario x_i, calcula P(x_i| y=0)
        # Guardalas en un arreglo de numpy:
        # self._likelihoods_negatives = [P(x_1| y=0), P(x_2| y=0), ..., P(x_n| y=0)]

        self._likelihoods_negatives = []
        for i in range(X.shape[1]):
            NegCount=np.sum((X[:,i]==0)&(y==0))
            NegLikelihood= (NegCount+self.alpha)/(np.sum(y==1)+2*self.alpha)
            self._likelihoods_negatives.append(NegLikelihood)
        self._likelihoods_negatives =  np.array(self._likelihoods_negatives)
        return self

    def predict(self, X):
        # TODO: Calcula la distribución posterior para la clase 1 dado los nuevos puntos X
        # utilizando el prior y los likelihoods calculados anteriormente
        # P(y = 1 | X) = P(y=1) * P(x1|y=1) * P(x2|y=1) * ... * P(xn|y=1)
        posterior_positive=1
        for i in range(X.shape[1]):
            if(X[0][i]==1):
                posterior_positive = self._likelihoods_positives[i]*posterior_positive
            else:
                posterior_positive=(1-self._likelihoods_positives[i])*posterior_positive
        # TODO: Calcula la distribución posterior para la clase 0 dado los nuevos puntos X
        # utilizando el prior y los likelihoods calculados anteriormente
        # P(y = 0 | X) = P(y=0) * P(x1|y=0) * P(x2|y=0) * ... * P(xn|y=0)
        posterior_negative=1
        for j in range(X.shape[1]):
            if(X[1][j]==1):
                posterior_negative = self._likelihoods_negatives[j]*posterior_negative
            else:
                posterior_negative = (1-self._likelihoods_negatives[j])* posterior_negative
        # TODO: Determina a que clase pertenece la muestra X dado las distribuciones posteriores
        clase = 
        return clase
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)