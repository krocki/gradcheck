# an example of numerical gradient check

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

<img src="AE.gif" width=100 />

`do_gradcheck` is set, all computation will be performed in float64

The error between the analytical solution (derivatives from backward) should be very close to the numerical one (less that 1e-6 err). Only a sample of parameters is going to be checked.

