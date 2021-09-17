import numpy as np
import matplotlib.pyplot as plt

N = 20
M = 15
y = np.array([i**2 + np.random.randint(0,15) for i in range(N)])
x = np.linspace(0, N-1, N)

y2 = np.array([i**2 + np.random.randint(0,3) for i in range(N, N-M, -1)])
x2 = np.linspace(N, N+M-1, M)

a, b = np.polyfit(x, y, deg=1)
y_est = a * x + b
y_err = x.std() * np.sqrt(1/len(x) +
                          (x - x.mean())**2 / np.sum((x - x.mean())**2))

a2, b2 = np.polyfit(x2, y2, deg=1)
y_est2 = a2 * x2 + b2
y_err2 = x2.std() * np.sqrt(1/len(x2) +
                          (x2 - x2.mean())**2 / np.sum((x2 - x2.mean())**2))

x = np.append(x, x2)
y = np.append(y, y2)
y_est = np.append(y_est, y_est2)
# y_err = np.append(y_err, y_err2)

# y_err = x.std() * np.sqrt(1/len(x) +
                        #   (x - x.mean())**2 / np.sum((x - x.mean())**2))

y_err = y.std() * np.sqrt(1/len(y) +
                          (y - y.mean())**2 / np.sum((y - y.mean())**2))


y_err2 = np.sqrt(1/len(y) * ((y - y.mean()) ** 2))

print("x: {}\ny: {}\ny_est: {}\ny_err: {}".format(x, y, y_est, y_err))
print("x: {}\ny: {}\ny_est: {}\ny_err: {}".format(len(x), len(y), len(y_est), len(y_err)))

# fig, ax = plt.subplots()
# ax.plot(x, y_est, '-')
# ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
# ax.plot(x, y, 'o', color='tab:green')
# plt.show()
# plt.close()

plt.plot(x, y_est, '-')
plt.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.4)
# plt.fill_between(x, y_est - y_err2, y_est + y_err2, alpha=0.2)
plt.plot(x, y, 'o', color='tab:green')
plt.show()
plt.close()
