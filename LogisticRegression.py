import numpy as np

def Sigmoide(z):
	return 1/(1 + np.exp(-z))

def Calc_Z(w, x, b):
    return np.dot(x, w.T) + b

def Cost_Function(y, sigm, examples):
	return (1/examples) * (np.sum(-(y * np.log(sigm) + (1 - y) * np.log(1 - sigm))))

def Gradient_Descent(sigm, w, b, y, x, examples, learning_rate):
	dz = sigm - y
	db = (1/examples) * np.sum(dz)
	dw = (1/examples) * x * dz
	dw = dw.sum(axis = 0)
	w = w - learning_rate * dw
	b = b - learning_rate *  db
	return w, b

def Init(examples, features, out = 1):
	w = np.random.randn(out, features)
	b = np.random.randn(examples, 1)
	return w, b

def Logistic_Regression(x, w, b, y, learning_rate, examples):
    for i in range(3):
        print("\nIteração {}:".format(i + 1))
        Z = Calc_Z(w, x, b)
        Sigm = Sigmoide(Z)
        J = Cost_Function(y, Sigm, examples)
        print("Cost Function: {}".format(J))
        w, b = Gradient_Descent(Sigm, w, b, y, x, examples, learning_rate)
        print("W: {}".format(w))
        print("B: {}".format(b))
        
    return w, b

X =  np.random.randn(5, 2)
W, B = Init(5, 2, 1)
Y = np.array([[0], [1], [0], [1], [0]])

print("After:")
print("X: {}".format(X))
print("W: {}".format(W))
print("B: {}".format(B))

W, B = Logistic_Regression(X, W, B, Y, 0.03, 5)

print("\nBefore:")
print("W: {}".format(W))
print("B: {}".format(B))
