import pandas as pd
import os
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.DataStructs.cDataStructs import ExplicitBitVect as ebv

os.remove('fingerprints_test.csv')

data = pd.read_csv('test.csv', usecols=['smiles'])
print data.head()
for i, row in data.iterrows():
  m = MolFromSmiles(row['smiles'])
  finger = RDKFingerprint(m)
  bits = ebv.ToBitString(finger)
  if len(bits) != 2048:
    print 'Uh oh...Row: {}, Smile: {}'.format(i, row['smiles'])
  output = ",".join(bits)
  with open("fingerprints_test.csv", "a") as myfile:
    myfile.write(row['smiles'])
    myfile.write(',')
    myfile.write(output)
    myfile.write('\n') 
  if i % 10000 == 0:
    print 'Finished {} smiles'.format(i)