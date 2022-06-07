
import openpyxl
import numpy as np
import statsmodels.api as sm
from sklearn import metrics
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

print('введіть шлях до файлу .xlsx')
dd = input()
def im(way, n):
    book = openpyxl.open(way, read_only=False)
    sheet = book.active
    ee = []
    for i in range(1,(n+1)):
        i1 = str(i)
        if n == 1:
            i1 = ''
        print('Введіть індекс стовбчика ' + i1 +' показника (A,B,C,ітд.) ')
        v = input()
        ee.append(v)
    R1 = input('Введіть початковий індекс строки (1,2,3...) ')
    R2 = input('Введіть кінцевий індекс строки(1,2,3...) ')
    R1 = int(R1)
    R2 = int(R2)
    e = []
    for r in range(R1,(R2+1)):
        for e1 in range(0,(n)):
            b = ee[e1]+ str(r)
            c = sheet[b].value
            e.append(c)
            #print(c)
    return(e)
    book.close()
print('введіть дані масиву кількісних показників')
print('введіть кількість кількісних показників')
aa=input()
aa = int(aa)
a = im(dd,aa)
#print(a)
x = np.array(a).reshape(-1,aa)
x = sm.add_constant(x)
#print(x)
print('Введіть дані масиву бінарного показника, поданого у вигляді 0 та 1')
b = im(dd,1)
y = np.array(b)
#print(y)
model = sm.Logit(y, x)
result = model.fit(method='newton')
ps = result.params
#print(ps)
#print(result.pred_table())
aa5 = result.predict(x)
#print(y)

bs = result.bse

'''
print(result.bse)
gh = float((len(a)))/float(aa)
gh = int(gh)
print(gh)
print(len(a))
for i in range(1, (gh+1)):
    print(i)
    itaa5 = round(aa5[i-1],0)
    if itaa5 == y[i-1]:
        print(1)
    else:
        print(0)
'''
#print(result.predict(x))

print(result.summary2())

print('коефіцієнт хі квадрат Вальда')
for i in range(1,(aa+2)):
    if i ==1:
        cons1 = ps[0]
        swc = bs[0]
        chiv = (cons1/swc)**2
        print('коефіцієнт хі квадрат Вальда для константи дорівнює',chiv)
    else :
        ie = i-1
        cons1 = ps[ie]
        swc = bs[ie]
        chiv = (cons1/swc)**2
        print('коефіцієнт хі квадрат Вальда для x',ie,' дорівнює',chiv)

fpr, tpr, _ = metrics.roc_curve(y,result.predict(x))

auc = metrics.roc_auc_score(y, aa5)
print('auc =', auc)
n1 = []
n2 = []

for i in range(1,(len(aa5)+1)):
    if y[i-1] == 0:
        n1.append(aa5[i-1])
    if y[i-1] != 0:
        n2.append(aa5[i-1])
#print(n1)
#print(n2)
U1, p = mannwhitneyu(n1, n2, method="exact")

print("Критерій Манна Уітні ", U1)
p = round(p,4)
if p<0.001:
    p1 = 'p<0.001'
else :
    p1 = str(p)
print("Рівень р для критерію Манна Уітні та AUC ", p1)

plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

input()
