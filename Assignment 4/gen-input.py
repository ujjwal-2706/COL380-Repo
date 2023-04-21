from argparse import ArgumentParser
import numpy as np

parser = ArgumentParser()
parser.add_argument("-n", type=int, required=True, help="Matrix size. n for an nxn matrix.")
parser.add_argument("-m", type=int, required=True, help="Block size. m for an mxm block.")
parser.add_argument("-o", dest="output", type=str, required=True, help="Output file name.")
args = parser.parse_args()

n = args.n
m = args.m
assert n % m == 0, "n mod(m) != 0"
np.random.seed(1)
k = n // m
print("n, m, k:", n,m,k)

l2 = np.random.choice(k**2, size=k, replace=False)
# print("len(l2):", len(l2))

l = [[i//k, i%k] for i in l2]
# l = [[i,j] for i,j in l if i<j]
l = np.array(l, dtype='int32')
# print("len(l):", len(l))

print()
print("n.to_bytes(4,'little'):", n.to_bytes(4,'little'))
print("m.to_bytes(4,'little'):",m.to_bytes(4,'little'))
print("k.to_bytes(4,'little'):",k.to_bytes(4,'little'))
print("len(l).to_bytes(4,'little'):", len(l).to_bytes(4,'little'))

with open(args.output, 'wb') as f:
    f.write(n.to_bytes(4,'little'))
    f.write(m.to_bytes(4,'little'))
    f.write(len(l).to_bytes(4,'little'))

    for x in l:
        p = np.random.randint(0,pow(2,15),size=(m,m), dtype = '<u2')
        f.write(x[0].tobytes())
        f.write(x[1].tobytes())
        f.write(p.tobytes())
