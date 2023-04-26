import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def toInt(data) :
    for coll in data.columns:
        if data[coll].dtype == 'object':
            data[coll] = data[coll].astype('category').cat.codes.astype(int)
    return data

# Load the data
training_data = pd.read_csv('train.csv')

# add to every na something
training_data[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = training_data[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)

training_data[["Deck", "Cabin_num", "Side"]] = training_data["Cabin"].str.split("/", expand=True)

#training_data['Destination'].fillna('TRAPPIST-1e', inplace=True)
#training_data['HomePlanet'].fillna('Earth', inplace=True)
training_data = toInt(training_data)


# Fill in missing values with mean
training_data.fillna(training_data.mean(), inplace=True)


y = training_data['Transported']
X = training_data.drop(['Transported', 'PassengerId'], axis=1)

training_data.head(10)
print(training_data[:10])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

rfc = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=100)
rfc.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rfc.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)




# --------------------------------------------------------------- #
testing_data = pd.read_csv('test.csv')
passenger_ids = testing_data['PassengerId']

testing_data['Destination'].fillna('TRAPPIST-1e', inplace=True)
testing_data[["Deck", "Cabin_num", "Side"]] = testing_data["Cabin"].str.split("/", expand=True)

test_data = toInt(testing_data)

# Fill in missing values with mean
testing_data.fillna(testing_data.mean(), inplace=True)

# Predict on the test data
X_test = testing_data.drop('PassengerId',axis=1)
y_pred = rfc.predict(X_test)

# Create a new DataFrame with PassengerId and predicted Transported values
output = pd.DataFrame({'PassengerId': passenger_ids, 'Transported': y_pred})

# Save the output to a new CSV file
output.to_csv('submission.csv', index=False)





