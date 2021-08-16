#In order for this code to run you need to download audi.csv file
#audi.csv file was downloaded from https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes page


import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

audi = pd.DataFrame(pd.read_csv('audi.csv'))
x,y = audi.shape

labels = list(audi.columns)
labels[-1],labels[2] = labels[2],labels[-1]
audi = audi.reindex(columns=labels)




Y = audi.price
X = audi[['year','engineSize','mileage','mpg']]

train_X, val_X, train_y, val_y = train_test_split(X,Y,random_state=0)

#function to calculate mean absolute error of prediction on some data, depending on number of leaf nodes in decision tree
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


#finding optimal value of number of leaf nodes which gets us lowest mean absolute error value
leaf_node_values = [val for val in range(2,1000)]
mae_values = {value:get_mae(value, train_X, val_X, train_y, val_y) for value in leaf_node_values}
min_leaf_nodes_value = min(mae_values, key=mae_values.get)
print(min_leaf_nodes_value)

#plotting results of calculating the lowest mean absolute error value
plt.plot(mae_values.keys(),list(mae_values.values()))
plt.scatter(min_leaf_nodes_value,mae_values.get(min_leaf_nodes_value))
plt.show()