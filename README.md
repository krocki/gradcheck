# Numerical gradient check in action

### STEP 0: get data 
```
./get_data.sh
```

A. MNIST classifier
```
python3 classify.py
```

B. Autoencoder
```
python3 ae.py
```

This is good, the err between numerical and analytical gradient is less than 1e-8.

```
iter   1000, loss = 422.25, gradcheck err 0.000000004 OK
iter   2000, loss = 335.09, gradcheck err 0.000000006 OK
iter   3000, loss = 395.82, gradcheck err 0.000000005 OK
iter   4000, loss = 307.89, gradcheck err 0.000000003 OK
iter   5000, loss = 291.37, gradcheck err 0.000000005 OK
iter   6000, loss = 251.52, gradcheck err 0.000000002 OK
iter   7000, loss = 239.43, gradcheck err 0.000000003 OK
```

<img src="AE.gif" width=100 />

`do_gradcheck` is set, all computation will be performed in float64

The error between the analytical solution (derivatives from backward) should be very close to the numerical one (less that 1e-6 err). Only a sample of parameters is going to be checked.

