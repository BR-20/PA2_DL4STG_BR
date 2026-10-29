d=64;N=7;T=96;tok=N*T;E=6;types=4;psi=8


def counts(features):
    emb=types*(features*d+d)+N*d
    S=(d*d+d)+types*d*d+((2*d+psi+1)*d+d)+d+2*d
    Tb=types*(3*d*d+d)+2*d
    head=(d+16+32)*128+128+128*3+3
    P=emb+5*(S+Tb)+36*32+32+head+6*16+7*2+6*2
    mac_S=tok*d*d+T*E*d*d+T*E*(2*d+psi+1)*d
    M=tok*features*d+5*(mac_S+tok*3*d*d)+4*6*((d+16+32)*128+128*3)
    return P,M


for name,features in (("main design",13),("calendar extension",17)):
    P,M=counts(features)
    print(name,"params",P,"MB32",round(P*4/1e6,2),"MACs",M,"GFLOP",round(2*M/1e9,3))
print("attn5",5*2*tok**2*d)
