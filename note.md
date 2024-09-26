avg the state dict by keys

env init: poisson -> init velocity -> init vehicle velocity
env update: vehicle position -> add/remove -> update velocity -> update vehicle states



# data rate

$$ \huge
S_{V_{i}^r} = B_{i, R_j} \cdot \log_2(1 + \frac{P_{R_j} \cdot h_i^r(V_{i}^r, R_j)}{\sigma^2})
$$

- $S_{V_{i}^r}$: the data rate of vehicle $V_{i}^r$
- $B_{i, R_j}$: the bandwidth of the channel between vehicle $V_{i}^r$ and RSU $R_j$
- $P_{R_j}$: the transmission power of RSU $R_j$
- $h_i^r(V_{i}^r, R_j)$: the channel gain between vehicle $V_{i}^r$ and RSU $R_j$
- $\sigma^2$: the noise power

# channel gain

$$ \huge
128.1 + 37.6 \cdot \log_{10}(d(V_{i}^r, R_j)) + X
$$

- $d(V_{i}^r, R_j)$: the distance between vehicle $V_{i}^r$ and RSU $R_j$
- $X$ (in dB): represents shadowing channel fading, which follows Log-normal distribution