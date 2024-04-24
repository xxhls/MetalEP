import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from share import ESM_PATH, PROT_PATH, LABELS_PATH

def do_pca(_X1, _X2, _n_components):
    pca1 = PCA(n_components=_n_components)
    pca2 = PCA(n_components=_n_components)
    pca1.fit(_X1)
    pca2.fit(_X2)
    _X1 = pca1.transform(_X1)
    _X2 = pca2.transform(_X2)
    return _X1, _X2

if __name__ == '__main__':
    
    esm = np.load(str(ESM_PATH))
    prot = np.load(str(PROT_PATH))
    labels = np.load(str(LABELS_PATH))
    print(esm.shape, prot.shape, labels.shape)
    
    X1 = esm
    X2 = prot
    y = labels
    
    X_pca = []
    print(X1.shape[0])
    for i in range(X1.shape[0]):
        print(f'Processing {i+1}/{X1.shape[0]}')
        x1 = X1[i]
        x2 = X2[i]
        x1_pca, x2_pca = do_pca(x1, x2, 100)
        x1_pca = x1_pca.flatten()
        x2_pca = x2_pca.flatten()
        X_pca.append(np.concatenate((x1_pca, x2_pca)))
    X = np.array(X_pca)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = svm.SVC(C=3, kernel='rbf', gamma='auto')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(roc_auc_score(y_test, y_pred))
    
