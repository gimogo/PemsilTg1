import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv', usecols=['NilaiA', 'NilaiB'])

x = df['NilaiA'].values.reshape(-1,1)
y = df['NilaiB'].values.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

lin_reg = LinearRegression()

lin_reg.fit(x_train, y_train)

print(lin_reg.coef_)
print(lin_reg.intercept_)

lin_reg.score(x_test, y_test)

y_prediksi = lin_reg.predict(x_test)
plt.scatter(x_test, y_test)
plt.plot(x_test, y_prediksi, c='r')
plt.xlabel('NilaiA')
plt.ylabel('NilaiB')
plt.title('Nilai A vs Nilai B')
plt.show()
