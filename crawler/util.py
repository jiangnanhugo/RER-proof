import inspect, random, sys

def sign( x ):
  """
  Returns 1 or -1 depending on the sign of x
  """
  if( x >= 0 ):
    return 1
  else:
    return -1

def raiseNotDefined():
  print "Method not implemented: %s" % inspect.stack()[1][3]    
  sys.exit(1)
  
class Counter(dict):
  """
  A counter keeps track of counts for a set of keys.
  
  The counter class is an extension of the standard python
  dictionary type.  It is specialized to have number values  
  (integers or floats), and includes a handful of additional
  functions to ease the task of counting data.  In particular, 
  all keys are defaulted to have value 0.  Using a dictionary:
  
  >>> a = {}
  >>> print a['test']
  
  would give an error, while the Counter class analogue:
    
  >>> a = Counter()
  >>> print a.getCount('test')
  0
  
  returns the default 0 value. Note that to reference a key 
  that you know is contained in the counter, 
  you can still use the dictionary syntax:
    
  >>> a = Counter()
  >>> a['test'] = 2
  >>> print a['test']
  2
  
  The counter also includes additional functionality useful for the
  classification project (later in the term).  Two counters can be added,
  subtracted or multiplied together.  See below for details.  They can
  also be normalized and their total count and arg max can be extracted.
  
  NOTE: You can check membership and size of a counter just like you can
  with a dictionary:

  >>> a = Counter()
  >>> a['test'] = 2
  >>> len(a)
  1
  >>> 'test' in a
  True
  >>> 'non-there' in a
  False
  """
  
  def incrementCount(self, key, count):
    """
    Increases the count of key by the specified count.  If 
    the counter does not contain the key, then the count for
    key will be set to count.
    
    >>> a = Counter()
    >>> a.incrementCount('test', 1)
    >>> a.getCount('hello')
    0
    >>> a.getCount('test')
    1
    """
    if key in self:
      self[key] += count
    else:
      self[key] = count
      
  def incrementAll(self, keys, count):
    """
    Increments all elements of keys by the same count.
    
    >>> a = Counter()
    >>> a.incrementAll(['one','two', 'three'], 1)
    >>> a.getCount('one')
    1
    >>> a.getCount('two')
    1
    """
    for key in keys:
      self.incrementCount(key, count)
      
  def setCount(self, key, count):
    """
    Sets the count of key to the specified count.
    """
    self[key] = count

  def getCount(self, key):
    """
    Returns the count of key, defaulting to zero.
    
    >>> a = Counter()
    >>> print a.getCount('test')
    0
    >>> a['test'] = 2
    >>> print a.getCount('test')
    2
    """
    if key in self:
      return self[key]
    else:
      return 0
  
  def argMax(self):
    """
    Returns some key with the highest value.
    """
    all = self.items()
    values = [x[1] for x in all]
    maxIndex = values.index(max(values))
    return all[maxIndex][0]


  def argMaxFair(self):
    """
    Returns a random key with the highest value.
    """
    all = self.items()
    random.Random().shuffle(all)
    values = [x[1] for x in all]
    maxIndex = values.index(max(values))
    return all[maxIndex][0]

#    all = self.items()
#    random.Random().shuffle(all)
#    return max(all, key = lambda x: x[1])[0]

  def getMax(self):
    """
    Returns the highest value.
    """
    return max(self.values())

  
  def sortedKeys(self):
    """
    Returns a list of keys sorted by their values.  Keys
    with the highest values will appear first.
    
    >>> a = Counter()
    >>> a['first'] = -2
    >>> a['second'] = 4
    >>> a['third'] = 1
    >>> a.sortedKeys()
    ['second', 'third', 'first']
    """
    sortedItems = self.items()
    compare = lambda x, y:  sign(y[1] - x[1])
    sortedItems.sort(cmp=compare)
    return [x[0] for x in sortedItems]
  
  def totalCount(self):
    """
    Returns the sum of counts for all keys.
    """
    return sum(self.values())
  
  def normalize(self):
    """
    Edits the counter such that the total count of all
    keys sums to 1.  The ratio of counts for all keys
    will remain the same. Note that normalizing an empty 
    Counter will result in an error.
    """
    total = float(self.totalCount())
    for key in self.keys():
      self[key] = self[key] / total
      
  def divideAll(self, divisor):
    """
    Divides all counts by divisor
    """
    divisor = float(divisor)
    for key in self:
      self[key] /= divisor

  def __mul__(self, y ):
    """
    Multiplying two counters gives the dot product of their vectors where
    each unique label is a vector element.
    
    >>> a = Counter()
    >>> b = Counter()
    >>> a['first'] = -2
    >>> a['second'] = 4
    >>> b['first'] = 3
    >>> b['second'] = 5
    >>> a['third'] = 1.5
    >>> a['fourth'] = 2.5
    >>> a * b
    14
    """
    sum = 0
    for key in self:
      if not (key in y):
        continue
      sum += self[key] * y[key]      
    return sum
      
  def __radd__(self, y):
    """
    Adding another counter to a counter increments the current counter
    by the values stored in the second counter.
    
    >>> a = Counter()
    >>> b = Counter()
    >>> a['first'] = -2
    >>> a['second'] = 4
    >>> b['first'] = 3
    >>> b['third'] = 1
    >>> a += b
    >>> a.getCount('first')
    1
    """ 
    for key, value in y.items():
      self.incrementCount(key, value)   
      
  def __add__( self, y ):
    """
    Adding two counters gives a counter with the union of all keys and
    counts of the second added to counts of the first.
    
    >>> a = Counter()
    >>> b = Counter()
    >>> a['first'] = -2
    >>> a['second'] = 4
    >>> b['first'] = 3
    >>> b['third'] = 1
    >>> (a + b).getCount('first')
    1
    """
    addend = Counter()
    for key in self:
      if key in y:
        addend[key] = self[key] + y[key]
      else:
        addend[key] = self[key]
    for key in y:
      if key in self:
        continue
      addend[key] = y[key]
    return addend
    
  def __sub__( self, y ):
    """
    Subtracting a counter from another gives a counter with the union of all keys and
    counts of the second subtracted from counts of the first.
    
    >>> a = Counter()
    >>> b = Counter()
    >>> a['first'] = -2
    >>> a['second'] = 4
    >>> b['first'] = 3
    >>> b['third'] = 1
    >>> (a - b).getCount('first')
    -5
    """      
    addend = Counter()
    for key in self:
      if key in y:
        addend[key] = self[key] - y[key]
      else:
        addend[key] = self[key]
    for key in y:
      if key in self:
        continue
      addend[key] = -1 * y[key]
    return addend
    