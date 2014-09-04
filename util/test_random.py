from tgen.planner import CandidateList
from tgen.tree import TreeData, NodeData
import random
import zlib

random.seed(1206)

l = CandidateList()
for i in xrange(10000):
#    l[str(i)] = random.randint(0, 100)
#    l[str(random.randint(0,1000))] = random.randint(0, 100)
#    l[(str(random.randint(0,1000)), str(random.randint(0,1000)))] = random.randint(0, 100)
#    tree = TreeData()
#    tree.create_child(0, 1, NodeData(str(random.randint(0, 1000)), str(random.randint(0, 1000))))
#    l[tree] = random.randint(0, 100)
    tree = TreeData()
    for j in xrange(random.randint(1,10)):
        tree.create_child(random.randint(0, len(tree)-1), 
                          random.randint(0, 1) == 1,
                          NodeData(str(random.randint(0, 1000)), str(random.randint(0, 1000))))
    l[tree] = random.randint(0, 100)
x = []
while l: x.append(l.pop())
print zlib.crc32(str(x))
