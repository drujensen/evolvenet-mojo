@value
struct Node:
  var synapes: List[Float64]

  fn __init__(inout self):
    self.synapes = List[Float64]()

fn main():
  var list: List[Node] = List[Node]()
  var list2: List[Node] = List[Node]()

  for i in range(10):
    var node = Node()
    node.synapes.append(i)
    list.append(node)

    var node2 = list[i]
    list2.append(node2)

  for i in range(10):
    print(list[i].synapes[0])
  for i in range(10):
    print(list2[i].synapes[0])

  list[0].synapes[0] = 100

  for i in range(10):
    print(list[i].synapes[0])
  for i in range(10):
    print(list2[i].synapes[0])
