from pandas import read_csv
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib


filename = "Iris.csv"
data = read_csv(filename)


print("Shape of the dataset:", data.shape)
print("First 20 rows:\n", data.head(20))


data.hist()
pyplot.savefig("histograms.png")
pyplot.close()  

data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.savefig("density_plots.png")
pyplot.close()


array = data.values
X = array[:, 1:5]  
Y = array[:, 5]   


test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)


result = model.score(X_test, Y_test)
print("Accuracy: {:.2f}%".format(result * 100))


joblib.dump(model, "logistic_model.pkl")
