import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import interactive
from numpy import linalg as LA
import cmath
import math
from fractions import Fraction as frac

one_list=  [ 1, 1 ]

def frac_part(num):
    numFrac_list = list(num)
    numFrac_list[0] = numFrac_list[0]%numFrac_list[1]
    return numFrac_list

def ret_num( num ):
    return num[0] / num[1]

def phase(x, y):                               #Converts a float array of length 2 to complex array of length 2
    xPhas = complex(0, x )
    yPhas = complex(0, y )
    return  [xPhas, yPhas]

def Suffix(num):                               #Generates proper suffix to ordinal number
    if num<0 or not isinstance(num,int):
        suffix = "!NO!"
    elif num>=10 and num<21:
        suffix="th"
    else:
        while(num>=10):
            num=int(num%10)
        if num == 1: suffix = "st"
        elif num == 2: suffix = "nd"
        elif num == 3: suffix = "rd"
        else: suffix = "th"
    return suffix

def substract_lists(list1, list2):
    list = [ 0 for j in range(2) ]
    list[1] = list1[1]*list2[1] /math.gcd(list1[1],list2[1])
    list[0] = list1[0]*list2[1]-list2[0]*list1[1] /math.gcd(list1[1],list2[1])
    return list

def mult_lists(list1, list2):
    list = [ 1 for j in range(2) ]
    list[1] = list1[1]*list2[1] /math.gcd(list1[1],list2[1])
    list[0] = list1[0]*list2[0]
    return list

def print_mat(mat, size):               # prints 2d array in a more pleasing manner
    for i in range(size):
        print(mat[i])

def indicate(num, alpha):                       #indicator function of interval 1-alpha to 1
    frac_alpha = frac_part(alpha)
    numtemp = frac_part(num)
    temp = substract_lists(one_list, frac_alpha)
    if 0<= numtemp[0] and numtemp[0] <temp[0]:
        return 0
    if numtemp[0]>=temp[0]:
        return 1

def fib(num):                                   #computes the numth fibonacci number
    if num<1 or type(num)!=int:
        print("Error in fib function.")
        return None
    if num == 1:
        return 2
    elif num == 2:
        return 3
    else:
        return fib(num-2)+fib(num-1)


def diag_operator(wid, alpha):
    size= wid**2
    mat = [[0 for j in range(size)] for i in range(size)]
    for i in range(size):
        for j in range(size):
            mat[i][j] = indicate( (i+j)*alpha, alpha )
    return np.array(mat)

def Op_Mat_NoPhase(wid, alpha):     #Generates the Schroedinger operator into a square matrix, wid denotes width of periodic patch
    size= wid**2             #size is the dimension of the operator matrix
    mat = [[0 for j in range(size)] for i in range(size)]
    for i in range(wid):
        for j in range(wid):
            if i != wid-1:
                mat[i*wid+j][(i+1)*wid+j] = 1
            if j != wid-1:
                mat[i*wid+j][i*wid+j+1] = 1
            if j != 0:
                mat[i*wid+j][i*wid+j-1] = 1
            if i != 0:
                mat[i*wid+j][(i-1)*wid+j] = 1
            mat[i*wid+j][i*wid+j] = Amp* indicate( mult_lists( frac_part(alpha), [i+1,1] ), alpha )
    return np.array(mat)

def potent_check(start, finish, alpha, l):
    alpha_num = ret_num(alpha)
    file_name= "Potential "+ str(l)+Suffix(l)+ " line check for Alpha=" \
    +str(round(alpha_num,4))+ ", from "+ str(start)+ " to "+  str(finish)
    f = open(file_name+".txt", "w")
    for j in range(start, finish+1):
        wid = fib(j)
        arr = [0 for i in range(wid)]
        for i in range(wid):
            arr[i] = indicate( mult_lists( frac_part(alpha), [i+1,1] ), alpha )
        f.write(str(arr)+"\n")
    f.close()



def Op_Mat_Phase(wid, phase):     #Generates the Schroedinger operator into a square matrix, wid denotes width of periodic patch
    size= wid**2            #size generates the dimension of the operator matrix
    mat = [[0 for j in range(size)] for i in range(size)]
    for j in range(wid):
        mat[0*wid+j][wid*(wid-1)+j] = cmath.exp( phase[1]    )
        mat[(wid-1)*wid+j][0*wid+j] =  cmath.exp(- phase[1])
    for i in range(wid):
        mat[i*wid+0][i*wid+ wid-1] = cmath.exp(- phase[0])
        mat[i*wid+ wid-1][i*wid+0] = cmath.exp( phase[0])
    return np.array(mat)

def sample_eig(wid, res, eig_num, alpha):
    x = np.linspace(-cmath.pi, cmath.pi, res + 1)
    y = np.linspace(-cmath.pi, cmath.pi, res + 1)
    size = wid ** 2
    if eig_num > 0:
        eig_mat = [[0 for a in range(res+1)] for b in range(res+1)  ]
    elif eig_num == 0:
        eig_mat = [[0.0 for a in range(size)] for s in range(2)]
    mat0 = Op_Mat_NoPhase(wid, alpha)
    for k in range(res+1):
        for l in range(res+1):
            mat1 = Op_Mat_Phase(wid, phase(x[k], y[l]))
            mat = np.add( mat0 , mat1  )
            vect=LA.eigvalsh( mat )
            if eig_num != 0:
                eig_mat[k][l] = vect[eig_num - 1]
            elif eig_num == 0:
                for  j in range(size):
                    if k==0 and l==0:
                        eig_mat[0][j] = vect[j]
                        eig_mat[1][j] = vect[j]
                    else:
                        eig_mat[0][j] = max(vect[j] , eig_mat[0][j] )
                        eig_mat[1][j] = min(vect[j], eig_mat[1][j])
            print( "Finished computing "+str(k*(res+1)+l+1) )
    return eig_mat

def print_band(size, mat, num):                  #print graph of spectral bands given matrix of maxima and minima
    x = [[0.0 for j in range(2)] for i in range(size)]
    y = [0.0 for i in range(size)]
    if num == 0:
        for i in range(size):
            x[i] = np.linspace(mat[1][i], mat[0][i], 3)
            y[i] = [i + 1 for j in range(3)]
            plt.plot(x[i], y[i])
        plt.show()
    elif num > 0 and num<=size:
        left = False
        right = False
        left_val = num
        right_val = num
        while not left:
            if left_val == 1:
                #left = True
                break
            left = ( mat[1][left_val-1] > mat[0][left_val-2])
            if left:
                break
            if not left:
                left_val = left_val-1
        while not right:
            if right_val == size:
                #right = True
                break
            right = ( mat[0][right_val-1] < mat[1][right_val])
            if right:
                break
            if not right:
                right_val = right_val+1
        for i in range(left_val-1, right_val):
            x[i] = np.linspace(mat[1][i], mat[0][i], 3)
            y[i] = [i + 1 for j in range(3)]
            plt.plot(x[i], y[i])
        plt.show()

def gen_spec_band(res, alpha, itera):
    alpha_num = ret_num(alpha)
    file_name="Fibonacci Spectral Bands " + str(itera)\
    +", Alpha="+str(round(alpha_num,3))+", Amplitude="+str(Amp)\
    +", Res="+str(res)
    wid = fib(itera)
    mat = sample_eig(wid, res, 0 , alpha)
    np.savetxt(file_name+".txt", mat, delimiter=",")

def print_tot_band(init_itera, fin_itera, res, alpha):
    alpha_num = ret_num(alpha)
    plt.title('Fibonacci Spectral Bands, '+ str(init_itera)+ \
              ' to '+ str(fin_itera)+", "+ "Alpha="\
              +str( round(alpha_num,3) )+", "\
              + "Amplitutde="+ str(Amp) )
    plt.xlabel('Spectrum values')
    plt.ylabel('Iteration number')
    for l in range(init_itera, fin_itera+1):
        wid = fib(l)
        size = wid**2
        mat = sample_eig(wid, res, 0, alpha)
        #np.savetxt("Fibo-alt-"+str(l)+".txt", mat ,delimiter=",")
        x = [[0.0 for j in range(2)] for i in range(size)]
        y = [0.0 for i in range(size)]
        for i in range(size):
            x[i] = np.linspace(mat[1][i], mat[0][i], 3)
            y[i] = [l for j in range(3)]
            plt.plot(x[i], y[i], color='green')
        print("Finished "+ str(wid)+Suffix(wid)+" matrix computation")
    plt.show()

def print_bands(init_itera, fin_itera, res, alpha):
    alpha_num = ret_num(alpha)
    plt.title('Fibonacci Spectral Bands, ' + str(init_itera) + \
              ' to ' + str(fin_itera) + ", " + "Alpha=" \
              + str(round(alpha_num, 3)) + ", " \
              + "Amplitutde=" + str(Amp)+", Res="+str(res))
    plt.xlabel('Spectrum values')
    plt.ylabel('Iteration number')
    for l in range(init_itera,fin_itera+1):
        read_name = "Fibonacci Spectral Bands " + str(l)\
        +", Alpha="+str(round(alpha_num,3))+", Amplitude="+str(Amp)\
        +", Res="+str(res)
        mat = np.loadtxt(read_name+".txt", delimiter=",")
        wid = fib(l)
        size = wid ** 2
        x = [[0.0 for j in range(2)] for i in range(size)]
        y = [0.0 for i in range(size)]
        for i in range(size):
            x[i] = np.linspace(mat[1][i], mat[0][i], 3)
            y[i] = [l+ 0*i/(size) for j in range(3)]
            plt.plot(x[i], y[i])
        print("Finished " + str(l) + Suffix(l) + " step plotting")
    plt.show()


def index_to_loc( ind, x, y):               # translated index in array to location on the x-y plane and returns string
    loca = np.array([0.0,0.0])
    loca[0] = x[ind[0]]
    loca[1] = y[ind[1]]
    stri = "("+ str(  round(loca[0] ,4) )+","+str( round(loca[1] ,4))+")"
    return stri

def plot_mat(mat, res, eig_num):
    x = np.linspace(-cmath.pi, cmath.pi, res + 1)
    y = np.linspace(-cmath.pi, cmath.pi, res + 1)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    if eig_num<0:
        colors = iter(cm.rainbow(np.linspace(-2, 2, -eig_num )))
    X, Y = np.meshgrid(x, y)
    if eig_num < 0:
        for j in range(-eig_num):
            Z= np.array(mat[j])
            ax.plot_wireframe(X, Y, mat[j],  label=str(j)+'-th eigenvalue.', color=next(colors))
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.legend(fontsize='small', loc=1)
            plt.title("Eigenvalues")
            plt.show()
    else:
        ax.plot_wireframe(X, Y, mat)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        #plt.legend([str(num) + '-th eigenvalue.'],  loc="upper left")
        ind_max =  np.unravel_index(np.argmax(mat, axis=None), mat.shape)
        ind_min =  np.unravel_index(np.argmin(mat, axis=None), mat.shape)
        plt.legend( [str( round( mat[ind_min],3)  )+ " to " + str( round(mat[ind_max],3) )] )
        plt.title(str(eig_num) +"-"+str(Suffix(eig_num))+ " eigenvalue")
        print("Maximal value is obtained in "+ index_to_loc( ind_max, x, y ) +" and is "\
              + str(round(mat[ind_max],4))+"." )
        print("Minimal value is obtained in " + index_to_loc( ind_min, x, y ) + " and is "\
              + str(round(mat[ind_min],4))+"." )
        plt.show()

alpha=(1+math.sqrt(5),2)
init_itera = 1
fin_itera = 7
itera = 2
res=10
Amp=10
wid = fib(fin_itera)
eig_num = 1
gen_spec_band(res, alpha, itera)
#mat = diag_operator(wid, alpha)
#print_mat( mat, wid**2 )
#mat = np.array(sample_eig(wid, res, eig_num, alpha))
#plot_mat(mat, res, eig_num)
#print_tot_band(init_itera, fin_itera, res, alpha)
#print_bands(init_itera, fin_itera, res, alpha)
#print( frac_part(alpha) )

#print( substract_lists(one_list, frac_part(alpha)) )
#val = (frac_part(alpha)*5<1-frac_part(alpha))
#print(val)
#val_list = mult_lists(frac_part(alpha),[7,1])
#print(indicate( val_list, alpha ))
#potent_check(init_itera, fin_itera , alpha, 1)
print("End of program test-run.")