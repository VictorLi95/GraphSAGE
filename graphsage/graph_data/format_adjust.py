fr = open('cora/features.features')
a = fr.readlines()
fr.close()

fw = open('cora/features.txt','w')
for i in range(len(a)):
  s = a[i].strip().split(' ')
  fw.write(s[0]+'\t')
  for j in range(1,len(s)):
    if (s[j]=='0.0'):
      fw.write('0\t')
    else:
      fw.write('1\t')
  fw.write('\n') 
fw.close()
